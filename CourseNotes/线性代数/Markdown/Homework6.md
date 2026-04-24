# Homework 6

## Ex.1 (T15)
**题目：** 设 $T \in \mathcal{L}(\mathbf{C}^3)$，使得 $6$ 和 $7$ 是 $T$ 的本征值，且 $T$ 关于 $\mathbf{C}^3$ 的任意基的矩阵都不是对角矩阵。证明存在 $(x,y,z) \in \mathbf{C}^3$ 使得
$$
T(x,y,z)=(17+8x,\sqrt{5}+8y,2\pi+8z).
$$

**证明：**
设 $v=(x,y,z)$，以及 $b=(17,\sqrt{5},2\pi)$。题目要证的等式等价于
$$
(T-8I)v=b.
$$
因此只需证明算子 $T-8I$ 是满射。

已知 $6,7$ 是 $T$ 的本征值。若 $T$ 的第三个本征值不是 $6$ 或 $7$，则 $T$ 有三个互异本征值，从而在 $\mathbf{C}^3$ 上可对角化，这将导致存在某组基使 $T$ 的矩阵是对角矩阵，与题设矛盾。
故 $T$ 的全部本征值只能是 $6,7$（重数允许重复）。

于是 $T-8I$ 的本征值只能是 $6-8=-2$ 与 $7-8=-1$，都不为 $0$，所以 $0$ 不是 $T-8I$ 的本征值。
这说明
$$
\ker(T-8I)=\{0\},
$$
即 $T-8I$ 是单射。由于定义域、值域同为有限维空间 $\mathbf{C}^3$，单射即满射，因此 $T-8I$ 是满射。

故对任意 $b\in\mathbf{C}^3$，都存在 $v\in\mathbf{C}^3$ 使 $(T-8I)v=b$。特别地，对 $b=(17,\sqrt{5},2\pi)$，存在 $(x,y,z)$ 满足
$$
T(x,y,z)=(17+8x,\sqrt{5}+8y,2\pi+8z).
$$
证毕。


## Ex.2 (T5)
**题目：** 设 $V$ 是有限维复向量空间且 $T\in\mathcal{L}(V)$。证明：$T$ 可对角化当且仅当对每个 $\lambda\in\mathbf{C}$ 都有
$$V=\text{null}(T-\lambda I)\oplus\text{range}(T-\lambda I)$$

**证明：**
记 $A_\lambda=T-\lambda I$。

先证必要性：若 $T$ 可对角化，则
$$
V=\bigoplus_{\mu\in\sigma(T)}E_\mu,
$$
其中 $E_\mu=\ker(T-\mu I)$。
固定 $\lambda\in\mathbf{C}$：
- $\text{null}(A_\lambda)=E_\lambda$（若 $\lambda\notin\sigma(T)$，则该空间为 $\{0\}$）；
- 对任意 $\mu\neq\lambda$，$A_\lambda$ 在 $E_\mu$ 上等于乘以非零标量 $(\mu-\lambda)$，故 $A_\lambda(E_\mu)=E_\mu$；在 $E_\lambda$ 上为零。
因此
$$
\text{range}(A_\lambda)=\bigoplus_{\mu\neq\lambda}E_\mu,
$$
从而
$$
V=E_\lambda\oplus\Big(\bigoplus_{\mu\neq\lambda}E_\mu\Big)=\text{null}(A_\lambda)\oplus\text{range}(A_\lambda).
$$
必要性成立。

再证充分性：假设对每个 $\lambda\in\mathbf{C}$ 都有
$$
V=\text{null}(A_\lambda)\oplus\text{range}(A_\lambda),
$$
特别地 $\text{null}(A_\lambda)\cap\text{range}  (A_\lambda)=\{0\}$。

取 $\lambda\in\sigma(T)$，证明其广义本征空间等于本征空间。设 $v\in\ker(A_\lambda^m)$（$m\ge1$），若 $v\notin\ker(A_\lambda)$，则存在最小 $j\ge2$ 使 $A_\lambda^j v=0$。
令 $u=A_\lambda^{j-2}v$，则
$$
A_\lambda u=A_\lambda^{j-1}v\neq0,
\qquad
A_\lambda^2u=A_\lambda^jv=0.
$$
于是 $A_\lambda u\in\ker(A_\lambda)$，且显然 $A_\lambda u\in\text{range}(A_\lambda)$，并且 $A_\lambda u\neq0$，这与
$$
\ker(A_\lambda)\cap\text{range}(A_\lambda)=\{0\}
$$
矛盾。
故必有 $\ker(A_\lambda^m)=\ker(A_\lambda)$，即每个本征值对应的 Jordan 块都只能是 $1\times1$。

因为在复数域上特征多项式可完全分解，$V$ 分解为各广义本征空间之直和；又每个广义本征空间等于本征空间，所以
$$
V=\bigoplus_{\lambda\in\sigma(T)}\ker(T-\lambda I),
$$
即 $T$ 可对角化。充分性成立。

综上，命题得证。


## Ex.3 (T18)
**题目：** 设 $V$ 是有限维复向量空间，$T\in\mathcal{L}(V)$。定义函数 $f:\mathbf{C}\to\mathbf{R}$ 为
$$
f(\lambda)=\dim\text{range}(T-\lambda I).
$$
证明 $f$ 不是连续函数。

**证明：**
设 $n=\dim V$（默认 $n\ge1$）。

对任意 $\lambda$，有
$$
f(\lambda)=\operatorname{rank}(T-\lambda I)\in\{0,1,\dots,n\},
$$
故 $f$ 只取整数值。

若 $\lambda$ 不是 $T$ 的本征值，则 $T-\lambda I$ 可逆，故
$$
f(\lambda)=n.
$$
若 $\lambda$ 是 $T$ 的本征值，则 $T-\lambda I$ 不可逆，故
$$
f(\lambda)<n.
$$

由于 $T$ 的本征值至多有限个（特征多项式的根），所以可取某个 $\mu\in\mathbf{C}$ 不是本征值，从而 $f(\mu)=n$；再取某个本征值 $\lambda_0$，有 $f(\lambda_0)<n$。因此 $f$ 不是常值函数。

另一方面，$\mathbf{C}$ 是连通集。若 $f$ 连续，则 $f(\mathbf{C})$ 必连通；但 $f(\mathbf{C})\subset\mathbf{Z}$，整数集在 $\mathbf{R}$ 中的连通子集只能是单点，故连续时 $f$ 必为常值函数。
这与上面得到的“$f$ 非常值”矛盾。

故 $f$ 不是连续函数。证毕。


## Ex.4 (T19)
**题目：** 设 $V$ 是有限维向量空间，$\dim V>1$，$T\in\mathcal{L}(V)$。证明
$$
\{p(T):p\in\mathcal{P}(\mathbf{F})\}\ne\mathcal{L}(V).
$$

**证明：**
设 $n=\dim V$，则 $n>1$。

由 Cayley-Hamilton 定理，$T$ 满足其特征多项式，从而任意多项式 $p$ 在 $T$ 处的值都可化为次数小于 $n$ 的多项式在 $T$ 处的值。故
$$
\{p(T):p\in\mathcal{P}(\mathbf{F})\}
\subset
\operatorname{span}\{I,T,T^2,\dots,T^{n-1}\}.
$$
因此
$$
\dim\{p(T):p\in\mathcal{P}(\mathbf{F})\}\le n.
$$

而
$$
\dim\mathcal{L}(V)=n^2.
$$
当 $n>1$ 时，$n^2>n$，所以一个维数至多为 $n$ 的子空间不可能等于维数为 $n^2$ 的整个空间 $\mathcal{L}(V)$。

故
$$
\{p(T):p\in\mathcal{P}(\mathbf{F})\}\ne\mathcal{L}(V).
$$
证毕。
