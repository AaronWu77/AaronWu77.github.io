# Homework 3
## Ex.1 (P77, T11)
**设 $v_1,\ldots,v_n$ 是 $V$ 中的向量，令$A=\{\lambda_1 v_1 + \cdots + \lambda_m v_m : \lambda_i \in F 且 \lambda_1+\cdots+\lambda_m=1\}$**
**(a)证明 $A$ 是 $V$ 的仿射子集**
**(b)证明 $V$ 的每个包含 $v_1,\ldots,v_n$ 的仿射子集都包含 $A$**
**(c)证明有某个 $v \in V$ 及 $V$ 的某个子空间 $U$ 使得 $A = v + U$ 且 $dimU \leq m-1$**

**证明:**
**(a)** 设 $x, y \in A$，则存在 $\lambda_1, \dots, \lambda_m \in F$ 且 $\sum_{i=1}^m \lambda_i = 1$，以及 $\mu_1, \dots, \mu_m \in F$ 且 $\sum_{i=1}^m \mu_i = 1$，使得 $x = \sum_{i=1}^m \lambda_i v_i$，$y = \sum_{i=1}^m \mu_i v_i$。
对于任意 $\alpha \in F$, 有 $\alpha x + (1-\alpha)y = \sum_{i=1}^m (\alpha \lambda_i + (1-\alpha)\mu_i)v_i$。
由于系数之和 $\sum_{i=1}^m (\alpha \lambda_i + (1-\alpha)\mu_i) = \alpha \sum_{i=1}^m \lambda_i + (1-\alpha)\sum_{i=1}^m \mu_i = \alpha \cdot 1 + (1-\alpha) \cdot 1 = 1$，
此时说明连线依然属于 $A$（如果考虑域是 $\mathbf{R}$ 或 $\mathbf{C}$，仿射子集即要求如此）。更一般的定义（子空间平移）: 取 $v = v_m$，易证 $A - v$ 是一个子空间，所以 $A$ 是平移后的子空间，即仿射子集。

**(b)** 设 $B$ 包含 $v_1, \ldots, v_m$ 且为仿射子集。我们将用数学归纳法（对 $m$）证明 $A \subseteq B$。
当 $m=1,2$ 时，由仿射子集的定义显然成立。
设对 $m-1$ 成立。对于 $x = \sum_{i=1}^m \lambda_i v_i \in A$（$\sum_{i=1}^m \lambda_i = 1$），若 $\lambda_m = 1$，则 $x=v_m \in B$；若 $\lambda_m \neq 1$，则
$x = (1-\lambda_m) \sum_{i=1}^{m-1} \frac{\lambda_i}{1-\lambda_m} v_i + \lambda_m v_m$。
由归纳假设，$\sum_{i=1}^{m-1} \frac{\lambda_i}{1-\lambda_m} v_i \in B$（其系数和为1）。又 $B$ 是仿射子集，故两点的仿射组合 $x \in B$。故 $A \subseteq B$。

**(c)** 取 $v = v_m \in A$。令 $U = \text{span}(v_1-v_m, \dots, v_{m-1}-v_m)$。
由于 $U$ 是由 $m-1$ 个向量生成的，因此是一个子空间，且 $\dim U \leq m-1$。
对任一 $x = \sum_{i=1}^m \lambda_i v_i \in A$，有 $\lambda_m = 1 - \sum_{i=1}^{m-1} \lambda_i$。
则 $x - v_m = \sum_{i=1}^{m-1} \lambda_i v_i + (1 - \sum_{i=1}^{m-1} \lambda_i)v_m - v_m = \sum_{i=1}^{m-1} \lambda_i (v_i - v_m)$。
这说明 $x - v \in U$，即 $A \subseteq v + U$。反向包含关系同样显然，故 $A = v + U$。

## Ex.2 (P77, T18)
**设 $T\in L(V,W)$， 并设 $U$ 是 $V$ 的一个子空间。用 $\pi$ 表示 $V$ 到 $V/U$ 的自然映射。证明存在 $S\in L(V/U,W)$ 使得 $T = S\circ \pi$ 当且仅当 $U \subseteq nullT$。**

**证明:**
$\Rightarrow$ (必要性): 假设存在 $S \in \mathcal{L}(V/U, W)$ 使得 $T = S \circ \pi$。
对于任意 $u \in U$，有 $\pi(u) = u + U = 0 + U$（即商空间 $V/U$ 中的零向量）。
那么 $T(u) = S(\pi(u)) = S(0 + U) = 0$。因此 $u \in \text{null } T$，故 $U \subseteq \text{null } T$。

$\Leftarrow$ (充分性): 假设 $U \subseteq \text{null } T$。
定义映射 $S: V/U \to W$ 为 $S(v + U) = T(v)$。
首先证明 $S$ 也是良定义的（well-defined）：对于同一个陪集，如果有 $v_1 + U = v_2 + U$，则 $v_1 - v_2 \in U \subseteq \text{null } T$。
从而 $T(v_1 - v_2) = 0 \implies T(v_1) = T(v_2)$。这说明 $S(v_1 + U) = S(v_2 + U)$，映射与代表元的选取无关。
接着验证 $S$ 是线性的：
$S((v_1 + U) + (v_2 + U)) = S((v_1 + v_2) + U) = T(v_1 + v_2) = T(v_1) + T(v_2) = S(v_1 + U) + S(v_2 + U)$；
$S(\lambda(v + U)) = S(\lambda v + U) = T(\lambda v) = \lambda T(v) = \lambda S(v + U)$。
因此 $S \in \mathcal{L}(V/U, W)$。
对于任意 $v \in V$，有 $(S \circ \pi)(v) = S(v + U) = T(v)$，所以 $T = S \circ \pi$。

## Ex.3 (P88, T6)
**设 $V$ 是有限维的，$v_1,\ldots,v_n$ 是 $V$ 中的向量。定义线性映射 $\Gamma : V'\rightarrow F^m$ 如下：**
$$\Gamma(\varphi) = (\varphi(v_1),\ldots,\varphi(v_m))$$
**(a) 证明 $v_1,\ldots,v_m$ 张成 $V$ 当且仅当 $\Gamma$ 是单的。**
**(b) 证明 $v_1,\ldots,v_m$ 是线性无关的当且仅当 $\Gamma$ 是满的。**

**证明:**
**(a)** $\Gamma$ 是单射当且仅当 $\text{null } \Gamma = \{0\}$。
对于 $\varphi \in V'$，$\varphi \in \text{null } \Gamma$ 当且仅当 $\Gamma(\varphi) = (\varphi(v_1), \dots, \varphi(v_m)) = (0, \dots, 0)$，即对于 $i=1,\dots,m$ 都有 $\varphi(v_i) = 0$。
这意味着 $\text{null } \Gamma = \text{span}(v_1, \dots, v_m)^0$（零化子）。
由零化子维数公式：$\dim V' = \dim \text{span}(v_1,\dots,v_m) + \dim (\text{span}(v_1,\dots,v_m)^0)$，且 $\dim V' = \dim V$。
因此 $\text{null } \Gamma = \{0\} \iff \dim (\text{span}(v_1,\dots,v_m)^0) = 0 \iff \dim \text{span}(v_1,\dots,v_m) = \dim V \iff \text{span}(v_1,\dots,v_m) = V$。
即 $v_1, \dots, v_m$ 张成 $V$ 当且仅当 $\Gamma$ 是单射。

**(b)** 由线性映射的基本定理，$\dim \text{range } \Gamma + \dim \text{null } \Gamma = \dim V' = \dim V$。
结合(a)中的结论 $\dim \text{null } \Gamma = \dim V - \dim \text{span}(v_1,\dots,v_m)$。
代入得：$\dim \text{range } \Gamma = \dim \text{span}(v_1,\dots,v_m)$。
$\Gamma$ 是满射当且仅当 $\text{range } \Gamma = F^m$，即 $\dim \text{range } \Gamma = m$。
所以，$\Gamma$ 是满射当且仅当 $\dim \text{span}(v_1,\dots,v_m) = m$，这就等价于 $v_1, \dots, v_m$ 是线性无关的（它们张成 $m$ 维空间）。

## Ex.4 (P88, T27)
**设 $T \in \mathcal{L}(\mathcal{P}_5(\mathbf{R}), \mathcal{P}_5(\mathbf{R}))$ 且 $\text{null } T' = \text{span}(\varphi)$，这里 $\varphi$ 是 $\mathcal{P}_5(\mathbf{R})$ 上的由 $\varphi(p) = p(8)$ 定义的线性泛函。证明 $\text{range } T = \{ p \in \mathcal{P}_5(\mathbf{R}) : p(8) = 0 \}$。**

**证明:**
回忆伴随映射基本性质：$(\text{range } T)^0 = \text{null } T'$。
题目中给出 $\text{null } T' = \text{span}(\varphi)$，因此 $(\text{range } T)^0 = \text{span}(\varphi)$。
令 $U = \{ p \in \mathcal{P}_5(\mathbf{R}) : p(8) = 0 \}$，我们要证明 $\text{range } T = U$。
注意对于任意 $p \in \mathcal{P}_5(\mathbf{R})$，$\varphi(p) = p(8) = 0$ 当且仅当 $p \in U$。所以 $U = \text{null } \varphi$。
对于任意 $q \in \text{range } T$，因为 $\varphi \in (\text{range } T)^0$，所以必定有 $\varphi(q) = 0$，即 $q(8) = 0$，从而 $q \in U$。这说明 $\text{range } T \subseteq U$。
再看维数，维数公式给出 $\dim \mathcal{P}_5(\mathbf{R}) = \dim \text{range } T + \dim (\text{range } T)^0$。
这里 $\dim \mathcal{P}_5(\mathbf{R}) = 6$（由 $1, x, x^2, x^3, x^4, x^5$ 张成），而 $\dim \text{span}(\varphi) = 1$（因为 $\varphi \neq 0$）。
所以 $\dim \text{range } T = 6 - 1 = 5$。
同时，泛函 $\varphi$ 是从 $\mathcal{P}_5(\mathbf{R})$ 到 $\mathbf{R}$ 的非零线性映射，所以其余核 $U = \text{null } \varphi$ 的维数为 $6 - 1 = 5$。
既然 $\text{range } T \subseteq U$ 并且两者的维数相等（皆为 5），因此必须有 $\text{range } T = U$。
即 $\text{range } T = \{ p \in \mathcal{P}_5(\mathbf{R}) : p(8) = 0 \}$。
