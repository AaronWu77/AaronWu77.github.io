# 第6章 内积空间

## 6.A 内积与范数
内积空间（inner product space）让我们可以讨论长度、角度和距离。

- `inner product` 记作 $\langle u,v\rangle$
- 满足共轭对称、线性、正定性
- `norm` 定义为
  $$
  \|v\|=\sqrt{\langle v,v\rangle}
  $$
- 由内积可以定义距离和正交

## 6.B 规范正交基
最好的基是既互相正交又都单位长度的基。

- `orthogonal`：两向量内积为 0
- `orthonormal`：正交且每个向量长度为 1
- `orthonormal basis` 让坐标表示和计算都更简单

## 6.C 正交投影与最小化
投影（projection）是内积空间里最实用的工具之一。

- 一个向量可以分解成“落在子空间上的部分”和“垂直部分”
- 最近点问题可以转化为正交投影
- 最小二乘（least squares）本质上也是投影问题

## 6.D Gram-Schmidt
Gram-Schmidt 过程可以把一组基改造成规范正交基。

- 先保持张成空间不变
- 再逐步去掉分量
- 最后归一化

## 6.E 直和与正交补
在内积空间中，正交结构会让分解更加自然。

- $U^\perp$：与 $U$ 中所有向量都正交的向量集合
- $V=U\oplus U^\perp$ 在很多情形下成立
- orthogonal complement 是后续谱理论的基础

## 6.F 本章作用
第6章把“向量空间”升级成可以测量的空间。

- 可以讨论长度、角度、距离
- 可以做投影和最小化
- 为 Chapter7 的正交算子和自伴算子做准备

