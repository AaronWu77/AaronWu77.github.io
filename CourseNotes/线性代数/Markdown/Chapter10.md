# 第10章 Trace 与 Determinant

## 10.A Trace
迹（trace）是矩阵和算子的一个重要不变量。

- `trace(T)` 通常定义为矩阵对角线元素之和
- 迹不依赖于选取的基
- 对角化后，trace 就是特征值之和

## 10.B Trace 的性质
迹有很多很好的代数性质。

- $\text{trace}(S+T)=\text{trace}(S)+\text{trace}(T)$
- $\text{trace}(cT)=c\,\text{trace}(T)$
- $\text{trace}(ST)=\text{trace}(TS)$
- 迹经常用来连接 operator 和 matrix

## 10.C Trace 与特征值
迹把特征值信息压缩成一个简单的数。

- 若 $T$ 可对角化，则 trace 等于所有特征值之和
- 迹与特征多项式、Jordan 结构有关
- 很多证明会先通过基变换把问题转回矩阵

## 10.D Determinant
行列式（determinant）描述的是线性映射对体积和可逆性的影响。

- $\det(T)=0$ 表示不可逆
- $\det(T)\neq 0$ 表示可逆
- 对角矩阵的 determinant 是对角线元素乘积

## 10.E Determinant 的性质
determinant 比 trace 更像“乘法型不变量”。

- $\det(ST)=\det(S)\det(T)$
- $\det(T^{-1})=1/\det(T)$（若可逆）
- 交换基以后 determinant 不变

## 10.F 迹与行列式的关系
这两者一起构成对算子最重要的两个数值不变量。

- trace 看“和”
- determinant 看“积”
- 它们都能从特征值读出
- 也都能帮助判断算子的结构
