# 第7章 内积空间上的算子

## 7.A 自伴算子
自伴算子（self-adjoint operator）是内积空间里最核心的一类算子。

- 若 $T=T^*$，则称 $T$ 为 self-adjoint
- 在实空间里，这对应对称矩阵的抽象版本
- 自伴算子具有非常好的特征值性质

## 7.B 正常算子与谱定理
正常算子（normal operator）满足和伴随算子相容的结构关系。

- 若 $TT^*=T^*T$，则 $T$ 是 normal
- 谱定理（spectral theorem）说明这类算子在合适基下有很好的标准形
- 对复空间，很多正常算子都可以由规范正交基对角化

## 7.C 正算子
正算子（positive operator）对应“不会产生负能量”的结构。

- 若 $\langle Tv,v\rangle \ge 0$，则 $T$ 是 positive
- 正算子与平方根、分解和优化问题有关
- 很多几何结论都能从正性推出

## 7.D 等距与酉 / 正交算子
保持长度不变的算子最像“刚体运动”。

- `isometry`：保持范数
- 在实空间中通常对应 orthogonal operator
- 在复空间中通常对应 unitary operator
- 这类算子保留角度和距离

## 7.E 极分解与奇异值分解
复杂算子常常可以拆成“旋转 + 拉伸”。

- `polar decomposition`：把算子写成酉/正交部分与正算子部分的组合
- `singular value decomposition (SVD)` 是更强的分解形式
- 这些分解在数值计算和数据分析里很重要

## 7.F 本章作用
第7章把内积空间中的结构彻底用起来。

- 自伴、normal、positive、isometry 是核心类型
- 谱定理提供标准形
- 分解理论连接了几何和代数

