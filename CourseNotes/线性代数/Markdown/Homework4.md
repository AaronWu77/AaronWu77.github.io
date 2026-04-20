# Homework 4

## Ex.1 (P100, T6)
设 $p\in \mathcal{P}(\mathbf{C})$ 的次数为 $m$。证明 $p$ 有 $m$ 个不同的零点当且仅当 $p$ 与其导式 $p'$ 没有公共零点。

**Proof:**
因为 $p$ 的次数为 $m$，根据代数基本定理，$p$ 在复数域内一定有 $m$ 个零点（按重数计算）。
因此，$p$ 有 $m$ 个不同的零点 $\iff$ $p$ 的所有零点都是单根（没有重根）。

若 $a$ 是 $p$ 的一个 $k$ 重根（$k \ge 1$），则可设 $p(z) = (z-a)^k q(z)$，其中 $q(a) \neq 0$。
对其求导得到：$p'(z) = k(z-a)^{k-1}q(z) + (z-a)^k q'(z)$。
- 如果 $k=1$（单根），则 $p'(a) = q(a) \neq 0$，即 $a$ 不是 $p'$ 的零点。
- 如果 $k \ge 2$（重根），由于两项均含 $(z-a)$ 因子，代入 $a$ 必有 $p'(a) = 0$。即 $a$ 同时是 $p$ 和 $p'$ 的公共零点。

综上所述，$p$ 没有重根 $\iff$ 没有任何根同时也是 $p'$ 的根（即 $p$ 和 $p'$ 没有公共零点）。得证。

## Ex.1 (P100, T8)
定义 $T: \mathcal{P}(\mathbf{R}) \to \mathbf{R}^{\mathbf{R}}$ 为
$$ T p = \begin{cases} \frac{p(x) - p(3)}{x - 3}, & \text{若 } x \neq 3, \\ p'(3), & \text{若 } x = 3. \end{cases} $$
证明对每个多项式 $p \in \mathcal{P}(\mathbf{R})$ 均有 $Tp \in \mathcal{P}(\mathbf{R})$, 且 $T$ 是线性映射。

**Proof:**
对于任意多项式 $p(x) \in \mathcal{P}(\mathbf{R})$，由因式定理（或多项式除法），总可以写成 $p(x) = (x-3)q(x) + p(3)$ 的形式，其中 $q \in \mathcal{P}(\mathbf{R})$ 也是实系数多项式。
因此当 $x \neq 3$ 时，分式即为：$\frac{p(x) - p(3)}{x-3} = q(x)$。
当 $x = 3$ 时，直接对 $p$ 展开式求导：$p'(x) = q(x) + (x-3)q'(x)$。将 $x=3$ 代入，得到 $p'(3) = q(3)$。
所以对所有的 $x \in \mathbf{R}$，都有 $(Tp)(x) = q(x)$。显然 $Tp = q \in \mathcal{P}(\mathbf{R})$，满足多项式条件。

接下来验证 $T$的线性性。任取两个多项式 $f, g$ 以及标量 $c$。
定义本身就是线性的组合操作（对多项式的差商及微积分操作在各个点上都是线性的）：
当 $x \neq 3$ 时，$\frac{(cf+g)(x)-(cf+g)(3)}{x-3} = c\frac{f(x)-f(3)}{x-3} + \frac{g(x)-g(3)}{x-3} = c (Tf)(x) + (Tg)(x)$。
当 $x = 3$ 时，$(cf+g)'(3) = c f'(3) + g'(3) = c(Tf)(3) + (Tg)(3)$。
因此 $T(cf+g) = c Tf + Tg$，即 $T$ 为线性映射。得证。

## Ex.1 (P100, T11)
设 $p \in \mathcal{P}(\mathbf{F})$, $p \neq 0$. 令 $U = \{pq : q \in \mathcal{P}(\mathbf{F})\}$.
(a) 证明 $\dim \mathcal{P}(\mathbf{F})/U = \deg p$.
(b) 求 $\mathcal{P}(\mathbf{F})/U$ 的一个基.

**Proof:**
(a) 设 $\deg p = m \ge 0$。根据多项式带余除法，对于任意 $f \in \mathcal{P}(\mathbf{F})$，存在唯一的商 $q \in \mathcal{P}(\mathbf{F})$ 和余式 $r$ 满足条件：
$f = pq + r$，其中 $\deg r < m$。
因为 $pq \in U$，所以在商空间中，每个多项式的等价类 $f + U$ 可以惟一地表示为余式的等价类 $r + U$。
这意味着商空间 $\mathcal{P}(\mathbf{F})/U$ 同构于所有次数小于 $m$ 的多项式构成的空间 $\mathcal{P}_{m-1}(\mathbf{F})$。
因此维数相等，$\dim \mathcal{P}(\mathbf{F})/U = \dim \mathcal{P}_{m-1}(\mathbf{F}) = m = \deg p$。得证。

(b) 由于对应着由于全体次数小于 $m$ 的多项式集合 $\mathcal{P}_{m-1}(\mathbf{F})$，它的一个自然基底是 $\{1, x, x^2, \ldots, x^{m-1}\}$。
对于其商空间来说也是一样的，故 $\mathcal{P}(\mathbf{F})/U$ 的一个基底为：
$$ \{ 1+U, \ x+U, \ x^2+U, \ \ldots, \ x^{m-1}+U \} $$
（当 $\deg p = 0$ 时，维数为 0，基为空集。）