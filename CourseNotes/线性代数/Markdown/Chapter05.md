# 第5章 特征值、特征向量与不变子空间

## 5.A 特征值与特征向量
特征值理论研究的是：一个线性映射是否存在“方向不变，只拉伸”的向量。

- 若 $Tv=\lambda v$，则 $\lambda$ 是 `eigenvalue`
- $v\neq 0$ 且满足上式，则 $v$ 是 `eigenvector`
- 所有对应同一特征值的向量加上零向量，构成 `eigenspace`
- 若 $T$ 有足够多的特征向量，就能把结构简化很多

## 5.B 不变子空间
不变子空间（invariant subspace）是理解特征值结构的另一种方式。

- 子空间 $U$ 若满足 $T(U)\subseteq U$，则称 $U$ 对 $T$ 不变
- 特征向量张成的空间总是不变的
- 不变子空间让我们把大问题拆成更小的问题

## 5.C 上三角化
很多时候不一定能直接对角化，但可以先变成上三角矩阵。

- 在合适的基下，$T$ 可表示成 upper-triangular matrix
- 对角线上的元素就是特征值
- 上三角形式让特征值、迹、行列式都更容易看清

## 5.D 对角化
对角化是最理想的简化形式。

- 若存在一组由特征向量组成的基，则 $T$ 可对角化
- 此时矩阵表示是 diagonal matrix
- 对角化以后，幂运算、函数运算和很多证明都更简单

## 5.E 本章结论
第5章的核心目标是把“算子”分解成更简单的代数对象。

- eigenvalue / eigenvector 提供局部结构
- invariant subspace 提供分块思路
- triangularization 和 diagonalization 提供计算工具

