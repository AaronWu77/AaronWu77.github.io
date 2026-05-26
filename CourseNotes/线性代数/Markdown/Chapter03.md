# 第3章 线性映射

## 3.A 线性映射
线性映射（linear map）是把一个向量空间中的结构“保持地”送到另一个向量空间中的函数。

- 设 $T:V\to W$
- 若对所有 $u,v\in V$ 和标量 $c\in F$ 都有
  - $T(u+v)=T(u)+T(v)$
  - $T(cv)=cT(v)$
  则称 $T$ 是线性映射
- 记所有从 $V$ 到 $W$ 的线性映射组成的集合为 $\mathcal{L}(V,W)$

### 常见例子

- 零映射（zero map）：把所有向量都送到 0
- 恒等映射（identity map）：$I(v)=v$
- 取值映射、微分算子、积分算子、矩阵乘法都可以看成线性映射

### 线性映射与定义域的基

设 $v_1,...,v_n$ 是 $V$ 的一个基，任意 $w_1,...,w_n\in W$，则存在唯一的线性映射 $T:V\to W$ 使得 $T(v_j)=w_j$ 对所有 $j=1,...,n$ 都成立。

### 线性映射的乘积

若 $T\in \mathcal{L}(U,V)$，$S\in \mathcal{L}(V,W)$，则 $S T\in \mathcal{L}(U,W)$ 定义如下：对于任意 $u\in U$，有 $(S T)(u) = S(T(u))$。

### 线性映射乘积的代数性质

- 结合性(associativity)：$(T_1T_2)T_3=T_1(T_2T_3)$
- 单位元(identity element)：$I T=T I=T$
- 分配律(distributivity)：$(S_1+S_2)T=S_1T+S_2T$ 和 $S(T_1+T_2)=ST_1+ST_2$，这里 $S, S_1, S_2\in \mathcal{L}(V,W)$，$T, T_1, T_2\in \mathcal{L}(U,V)$

## 3.B 零空间与值域

### 零空间 (Null Space)
线性映射 $T$ 的零空间定义为

$$
\operatorname{null}T=\{v\in V:T(v)=0\}
$$

- $\operatorname{null}T$ 是 $V$ 的子空间
- 若 $T(v_1)=T(v_2)$，则 $v_1-v_2\in \operatorname{null}T$
- 若 $\operatorname{null}T=\{0\}$，则 $T$ 是 one-to-one

## 单的 (injective)

如果当 $Tu=Tv$ 时必有 $u=v$，则称 $T$ 是单的（one-to-one 或 injective）。
同时，单射性等价于零空间为 ${0}$

### 值域 (Range)

线性映射 $T$ 的值域定义为

$$
\operatorname{range}T=\{T(v):v\in V\}
$$

- $\operatorname{range}T$ 是 $W$ 的子空间
- 若 $\operatorname{range}T=W$，则 $T$ 是 满的 (onto)
- 核和像分别描述“丢失了什么”和“到达了哪里”

### 线性映射基本定理
- 设 $V$ 是有限维的， $T\in\mathcal{L}(V,W)$，则 $\operatorname{range}T$ 是有限维的并且 $dim V = dim \operatorname{null}T + dim \operatorname{range}T$。
- 如果 $V$ 和 $W$ 是有限维的，并且 $\dim V > \dim W$，则 $T$ 不是单的。
- 如果 $V$ 和 $W$ 是有限维的，并且 $\dim V < \dim W$，则 $T$ 不是满的。
- 齐次线性方程组：当变量多于方程时，齐次线性方程组必有非零解
- 非齐次线性方程组：当方程多余变量时，必有一组常数项使得对应的非齐次线性方程组无解

### Nullity and Rank

- $\operatorname{nullity}T=\dim(\operatorname{null}T)$
- $\operatorname{rank}T=\dim(\operatorname{range}T)$

## 3.C 矩阵
一旦选定基，线性映射就可以用矩阵表示。

- 设 $V$ 的基为 $(v_1,\dots,v_n)$，$W$ 的基为 $(w_1,\dots,w_m)$
- 对每个 $j$，把 $T(v_j)$ 写成 $W$ 中基的线性组合
- 这些坐标列拼起来，就得到 $T$ 的矩阵表示

### 矩阵的意义

- 矩阵的第 $j$ 列就是 $T(v_j)$ 的坐标
- 对任意 $v\in V$，都有
  $$
  [T(v)]_W=M(T)[v]_V
  $$
- 这说明“算子作用”可以转成“矩阵乘法”

### 组合与乘法

- 若先做 $T$ 再做 $S$，则对应矩阵相乘
- 也就是
  $$
  M(S\circ T)=M(S)M(T)
  $$
- 所以矩阵乘法本质上是在编码线性映射的复合

## 3.D 可逆线性映射
线性映射如果能被反向恢复，就说明它没有丢信息。

- 若存在 $S:W\to V$ 使得 $S\circ T=I_V$ 且 $T\circ S=I_W$，则称 $T$ 可逆（invertible）
- 可逆时，$S$ 就是 $T$ 的 inverse

### 等价条件

对于有限维情形，下面常常等价：

- $T$ 可逆
- $T$ 是 one-to-one
- $T$ 是 onto
- $\operatorname{null}T=\{0\}$
- $\operatorname{range}T=W$

### Isomorphism

- 可逆的线性映射也叫 `isomorphism`
- 若两个向量空间之间存在 isomorphism，则它们在结构上是“同一种空间”

## 3.E 换基与相似
同一个线性映射，在不同基下会有不同矩阵表示。

- 线性映射本身不变
- 矩阵会随基变化而变化
- 表示同一个算子的不同矩阵彼此 `similar`

这也是为什么后面研究 eigenvalue 时，必须区分“算子本身”与“它的矩阵表示”。

## 3.F Rank-Nullity Theorem
有限维线性代数中最重要的结论之一就是秩-零化度定理。

$$
\dim(\operatorname{null}T)+\dim(\operatorname{range}T)=\dim V
$$

也就是

- `nullity(T) + rank(T) = dim(V)`

### 直观理解

- 核描述了输入中被“压没”的部分
- 像描述了输出中真正保留下来的部分
- 两者加起来，正好是输入空间的维数

## 3.G 本章作用
第3章把向量空间的内容推进到“映射”层面。

- 线性映射是后续所有章节的核心对象
- 核、像、秩、零化度是分析结构的基本工具
- 矩阵是线性映射的计算表达
- 可逆性和同构为后面的特征值、对角化打基础

