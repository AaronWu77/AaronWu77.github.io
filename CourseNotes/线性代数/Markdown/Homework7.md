# Homework 7

## Ex.1 (T8)
**题目：** 设 $V$ 是有限维的，$P\in\mathcal{L}(V)$ 使得 $P^2=P$ 且对每个 $v\in V$ 均有 $\|Pv\|\le\|v\|$。证明存在 $V$ 的子空间 $U$ 使得 $P=P_U$。

**证明：**
令
$$
U=\operatorname{range}(P).
$$
由 $P^2=P$ 可知对任意 $v\in V$，
$$
v=Pv+(v-Pv),\qquad Pv\in U,\quad v-Pv\in\ker P,
$$
且 $U\cap\ker P=\{0\}$，故
$$
V=U\oplus\ker P.
$$

下证 $\ker P\subset U^\perp$。任取 $u\in U$、$w\in\ker P$。由 $u\in U$，存在 $x\in V$ 使 $u=Px$。对任意标量 $t$，有
$$
P(u+tw)=Pu+tPw=u,
$$
于是由条件 $\|Pz\|\le\|z\|$（取 $z=u+tw$）得
$$
\|u\|\le \|u+tw\|\qquad(\forall t).
$$
即函数 $\phi(t)=\|u+tw\|^2$ 在 $t=0$ 处取最小值。展开内积：
$$
\phi(t)=\|u\|^2+2\operatorname{Re}\big(t\langle w,u\rangle\big)+|t|^2\|w\|^2.
$$
取 $t\in\mathbf{R}$ 得 $\operatorname{Re}\langle w,u\rangle=0$；再取 $t\in i\mathbf{R}$ 得 $\operatorname{Im}\langle w,u\rangle=0$。故
$$
\langle w,u\rangle=0,
$$
从而 $w\perp u$。因此 $\ker P\subset U^\perp$。

又由维数公式：
$$
\dim\ker P=\dim V-\dim U=\dim U^\perp,
$$
故包含关系必为等号：
$$
\ker P=U^\perp.
$$
于是对任意 $v\in V$，分解
$$
v=Pv+(v-Pv),\quad Pv\in U,\ (v-Pv)\in U^\perp,
$$
这正是到 $U$ 的正交投影分解，所以 $P=P_U$。证毕。


## Ex.2 (T14)
**题目：** 设 $C_R([-1,1])$ 是区间 $[-1,1]$ 上实值连续函数构成的向量空间，且其上的内积为
$$
\langle f,g\rangle=\int_{-1}^1 f(x)g(x)\,dx.
$$
给定 $C_R([-1,1])$ 的子空间
$$
U=\{f\in C_R([-1,1]):f(0)=0\}.
$$
(a) 证明 $U^\perp=\{0\}$。

(b) 证明当没有“有限维”这一假设时，6.47 和 6.51 不成立。

**证明：**
(a) 设 $g\in U^\perp$，即
$$
\int_{-1}^1 g(x)f(x)\,dx=0\qquad(\forall f\in U).
$$
任取 $h\in C_R([-1,1])$，定义
$$
f_n(x)=h(x)\big(1-e^{-nx^2}\big).
$$
则 $f_n\in C_R([-1,1])$ 且 $f_n(0)=0$，故 $f_n\in U$，从而
$$
\int_{-1}^1 g(x)f_n(x)\,dx=0.
$$
当 $n\to\infty$ 时，$f_n(x)\to h(x)$（逐点），且
$$
|g(x)f_n(x)|\le |g(x)h(x)|,
$$
右侧可积。由控制收敛定理，
$$
0=\lim_{n\to\infty}\int_{-1}^1 g(x)f_n(x)\,dx
=\int_{-1}^1 g(x)h(x)\,dx.
$$
故对一切 $h\in C_R([-1,1])$ 都有 $\int gh=0$。取 $h=g$，得
$$
\int_{-1}^1 g(x)^2\,dx=0,
$$
于是 $g\equiv0$。所以
$$
U^\perp=\{0\}.
$$

(b) 本题空间 $V=C_R([-1,1])$ 是无限维空间。由 (a) 已知 $U^\perp=\{0\}$，于是
$$
U\oplus U^\perp=U\ne V
$$
（例如常函数 $1\notin U$）。因此“$V=U\oplus U^\perp$”在此不成立，这说明 6.47 在无限维情形下失效。

又
$$
U^{\perp\perp}=(\{0\})^\perp=V\ne U,
$$
故“$U^{\perp\perp}=U$”也不成立，这说明 6.51 在无限维情形下失效。证毕。


## Ex.3 (T14)
**题目：** 设 $e_1,\dots,e_n$ 是 $V$ 的规范正交基，并设 $v_1,\dots,v_n$ 是 $V$ 中的向量，使得对每个 $j$ 都有
$$
\|e_j-v_j\|<\frac{1}{\sqrt n}.
$$
证明 $v_1,\dots,v_n$ 是 $V$ 的基。

**证明：**
定义线性算子 $A\in\mathcal L(V)$ 使
$$
Ae_j=v_j\quad(j=1,\dots,n),
$$
再令
$$
T=I-A.
$$
则
$$
Te_j=e_j-v_j,
\qquad
\|Te_j\|<\frac1{\sqrt n}.
$$
设
$$
\alpha=\max_{1\le j\le n}\|Te_j\|<\frac1{\sqrt n}.
$$
任取 $x=\sum_{j=1}^n a_je_j$，有
$$
Tx=\sum_{j=1}^n a_jTe_j,
$$
故
$$
\|Tx\|
\le \sum_{j=1}^n |a_j|\,\|Te_j\|
\le \alpha\sum_{j=1}^n |a_j|
\le \alpha\sqrt n\Big(\sum_{j=1}^n |a_j|^2\Big)^{1/2}
=\alpha\sqrt n\,\|x\|.
$$
记 $c=\alpha\sqrt n<1$，则
$$
\|Tx\|\le c\|x\|\qquad(\forall x\in V).
$$

若 $Ax=0$，则 $x=Tx$，从而
$$
\|x\|=\|Tx\|\le c\|x\|.
$$
因 $c<1$，只能有 $x=0$。所以 $A$ 单射。有限维下单射即满射，故 $A$ 可逆。

于是
$$
(v_1,\dots,v_n)=(Ae_1,\dots,Ae_n)
$$
是 $V$ 的一组基。证毕。


## Ex.4 (T8)
**题目：** 求多项式 $q\in P_2(\mathbf R)$，使得对每个 $p\in P_2(\mathbf R)$ 均有
$$
\int_0^1 p(x)\cos(\pi x)\,dx=\int_0^1 p(x)q(x)\,dx.
$$

**证明：**
设
$$
q(x)=ax^2+bx+c.
$$
题设等价于
$$
\int_0^1 p(x)\big(\cos(\pi x)-q(x)\big)\,dx=0\qquad(\forall p\in P_2).
$$
取基 $1,x,x^2$，得到方程组：
$$
\int_0^1 q(x)\,dx=\int_0^1\cos(\pi x)\,dx=0,
$$
$$
\int_0^1 xq(x)\,dx=\int_0^1 x\cos(\pi x)\,dx=-\frac2{\pi^2},
$$
$$
\int_0^1 x^2q(x)\,dx=\int_0^1 x^2\cos(\pi x)\,dx=-\frac2{\pi^2}.
$$
即
$$
\frac a3+\frac b2+c=0,
$$
$$
\frac a4+\frac b3+\frac c2=-\frac2{\pi^2},
$$
$$
\frac a5+\frac b4+\frac c3=-\frac2{\pi^2}.
$$
解得
$$
a=0,\qquad b=-\frac{24}{\pi^2},\qquad c=\frac{12}{\pi^2}.
$$
故
$$
q(x)=\frac{12}{\pi^2}(1-2x).
$$
该解满足题设（且由内积非退化性可知解唯一）。证毕。
