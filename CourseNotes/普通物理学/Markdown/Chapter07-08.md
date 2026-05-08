# Chapter 7 Systems of Particles

## 质心的计算

多质子系统的质心求解方法从两个质子的系统推广而来

位置：
$$\begin{align}
\vec r_{cm}&=\frac{m_1\vec r_1+m_2\vec r_2+\cdots + m_N\vec r_N}{m_1+m_2+\cdots +m_N} = \frac{1}{M}\sum m_i \vec r_i\\
x_{cm}&=\frac{1}{M}\sum m_i x_i\\
y_{cm}&=\frac{1}{M}\sum m_i y_i\\
z_{cm}&=\frac{1}{M}\sum m_i z_i\\
\end{align}
$$

速度和加速度：
$$\begin{align}
\vec v_{cm}&=\frac{1}{M}\sum m_i \vec v_i\\
\vec a_{cm}&=\frac{1}{M}\sum m_i \vec a_i\\
\end{align}
$$

对于一个实心的物体 (solid body) 来说，求解质心的方法需要使用微分和积分，即

$$\vec r_{cm} = \frac{1}{M}\int \vec r dm\\
x_{cm}=\frac{1}{M}\int xdm\\
y_{cm}=\frac{1}{M}\int ydm\\
z_{cm}=\frac{1}{M}\int zdm\\
$$

---
## 对于不同类型的刚体质心计算

- 体分布：$\rho$ 表示这个物体的密度，单位体积的质量，则 $dm = \rho dV$，其中 $dV$ 是一个体积元素
- 面分布：$\sigma$ 表示这个物体的面密度，单位面积的质量，则 $dm = \sigma dA$，其中 $dA$ 是一个面积元素
- 线分布：$\lambda$ 表示这个物体的线密度，单位长度的质量，则 $dm = \lambda dl$，其中 $dl$ 是一个长度元素

**Example 1**:一个长度为L的杆子，其线密度为 $\lambda = ax$，其中 $x$ 是从杆子的一端到质心的距离，求这个杆子的质心位置。
$$\begin{align}
x_{cm}&=\frac{\int_0^L x dm}{\int_0^L dm}\\
&=\frac{\int_0^L x \lambda dl}{\int_0^L \lambda dl}\\
&=\frac{\int_0^L x (ax) dx}{\int_0^L ax dx}\\
&=\frac{\int_0^L ax^2 dx}{\int_0^L ax dx}\\
&=\frac{aL^3/3}{aL^2/2} = \frac{2L}{3}
\end{align}$$

**Example 2**:一个半径为R的半圆板（设直径在X轴上，圆弧在X轴上方）。 其面积密度为 $\rho$,求解这个半圆办板的质心的位置。

取一条水平的窄条，在$\theta$处，宽度为$2R\cos\theta$,高度为$dy=R\cos\theta d\theta$,则这个窄条的面积为$ds=2R^2\cos^2\theta d\theta$, 质量为$dm=\rho ds=2\rho R^2\cos^2\theta d\theta$,位置为$(R\cos\theta, R\sin\theta)$,则
$$
\begin{align}
y_{cm}&=\frac{\int y dm}{\int dm}\\
&=\frac{\int_0^{\pi} R\sin\theta (2\rho R^2\cos^2\theta d\theta)}{\int_0^{\pi} 2\rho R^2\cos^2\theta d\theta}\\
&=\frac{2\rho R^3 \int_0^{\pi} \sin\theta \cos^2\theta d\theta}{2\rho R^2 \int_0^{\pi} \cos^2\theta d\theta}\\
&=\frac{2\rho R^3 \cdot 1/3}{2\rho R^2 \cdot \pi/4} = \frac{4R}{3\pi}
\end{align}
$$

---
## 质心运动定理

- 质点系内各个质点受外力和内力，内力成对出现，矢量和为0，所以：$\sum \vec F_{ext} = \sum m_i\vec a_i = M \vec a_{cm}$
- 置信的运动等同于一个质点，其质量等于总质量，受所有外力的合力 $\sum F_{ext,x}=M a_{cm,x}$, $\sum F_{ext,y}=M a_{cm,y}$, $\sum F_{ext,z}=M a_{cm,z}$

根据上述的运动定理，我们不难得出关于质心运动的动量守恒定律

---
## 质心动量守恒定律

- 系统总动量：$\vec P = M \vec v_{cm}$
- 牛顿第二定律：$\sum \vec F_{ext} = M \vec a_{cm} = \frac{d\vec P}{dt}$

---
## 质心参考系

以质心为原点的参考系称为质心系，在其中
- 质心位置：$\vec r'_{cm} = 0$
- 各个质点相对于质心的位置矢量，速度，加速度：
  - $\vec r'_i = \vec r_i - \vec r_{cm}$
  - $\vec v'_i = \vec v_i - \vec v_{cm}$
  - $\vec a'_i = \vec a_i - \vec a_{cm}$
- 质心系中动量守恒定律：$\sum \vec F'_{ext} = \sum m_i \vec a'_i = 0$


---
## 变质量运动方程

当主体质量随着时间变化（比如火箭喷气等等），需考虑动量变化。
假设主体 t 时刻的质量为 M, 速度为 $\vec v$, dt 时间内并入或抛出质量 $dM$, 相对主体的速度为$\vec v_{rel}= \vec u-\vec v$,($\vec u$ 为质量元的绝对速度)，则动量变化为
$$ M\frac{d\vec v}{dt} = \sum \vec F_{ext} + \vec v_{rel} \frac{dM}{dt}$$  

---

### 火箭方程（特例推导）

若火箭在外太空无重力空间下飞行：$\sum \vec{F}_{\text{ext}} = 0$

喷气相对速度大小为 $u$，方向向后：$\vec{v}_{\text{rel}} \cdot d\vec{v}/dt$ 在一维标量下写为 $-u$。而 $dM/dt = -R$（$R$ 为质量消耗率，为正值）。则方程：

$$
M \frac{dv}{dt} = 0 + (-u) \cdot (-R) = uR
$$

由于 $R = -\frac{dM}{dt}$，代入得：

$$
M \frac{dv}{dt} = -u \frac{dM}{dt}
$$

两边乘以 $dt$ 并整理：

$$
dv = -u \frac{dM}{M}
$$

积分：$\int_{v_i}^{v_f} dv = -u \int_{M_i}^{M_f} \frac{dM}{M}$

得到著名的**齐奥尔科夫斯基基火箭方程**：

$$
v_f - v_i = u \ln \left( \frac{M_i}{M_f} \right)
$$

此式直接给出了火箭在不受外力，仅靠喷射工质加速所能达到的速度增量。

# Chapter 8 Rotational Kinematics

## 基本概念

- 角位移 $\phi$：描述物体转过的角度，通常以弧度 (rad) 为单位
- 角速度 $\omega$：描述物体转动的快慢，定义为角位移对时间的导数，即 $\omega = \frac{d\phi}{dt}$，单位为弧度每秒 (rad/s)
- 角加速度 $\alpha$：描述角速度变化的快慢，定义为角速度对时间的导数，即 $\alpha = \frac{d\omega}{dt}$，单位为弧度每秒平方 (rad/s²)
- 匀变角速度公式：
  - $\omega=\omega_0+\alpha t$
  - $\phi=\phi_0+\omega_0 t + \frac{1}{2} \alpha t^2$
  - $\omega^2 = \omega_0^2 + 2\alpha (\phi-\phi_0)$

---
## 线量与角量的关系

转动的刚体上的每个点都在做圆周运动。设某点离转轴距离为r

- 路程为角位移：$s=r\phi$, $ds=rd\phi$
- 线速度与角速度：$v=\frac{ds}{dt}=r\frac{d\phi}{dt}=r\omega$
- 加速度的两个分量：
  - 切向加速度：$a_t=r\alpha$
  - 向心加速度：$a_c=\frac{v^2}{r}=r\omega^2$
  - 总加速度：$a=\sqrt{a_t^2+a_c^2}$

> 这两章的内容主要还是从习题的角度出发