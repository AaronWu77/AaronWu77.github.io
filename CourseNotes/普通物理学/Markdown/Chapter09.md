# Chapter 9 Rotational Dynamics

## 力矩 (Torque)

力矩是使物体发生转动的的原因，对应平动运动中的力F
- 力矩矢量表达形式：$\vec \tau = \vec r \times \vec F$
- 力矩标量表达形式：$\tau = r F \sin \theta$，其中 $\theta$ 是 $\vec r$ 和 $\vec F$ 之间的夹角
- $r\sin \theta$ 是力臂，表示力的作用线与转轴之间的距离

![medium left](PIC/Chapter9-1.png)

从上方图片中我们可以看到关于力矩方向的确定，根据右手定则：如果 $\vec r$ 和 $\vec F$ 的旋转方向是逆时针的，那么 $\vec \tau$ 的方向是垂直于平面向上的；如果是顺时针的，那么 $\vec \tau$ 的方向是垂直于平面向下的。

---
## 转动惯量 (Rotational Inertia)

转动惯量是物体转动惯性的大小，对应平动里的质量m，描述物体抵抗角加速度的能力
- 单个质点：$I=mr^2$
- 质点系：$I=\sum m_ir_i^2$
- 连续刚体：$I=\int r^2dm$

从上面的公式中可以看出，转动惯量取决于质量分布和转轴的位置
- 质量距离转轴越远，转动惯量越大
- 同一刚体，转轴不同I也不同

> 根据上述的概念以及在平动里面提到的 F=ma，我们不难推出转动惯量和力矩之间的关系 $ \vec \tau = I \vec \alpha $ (其中 $\vec \alpha$ 是角加速度)。

![medium left](PIC/Chapter9-2.png)

接下来我们来分析上面这一页PPT的具体内容
- 有一个外力 $F$ 作用在当前物体上，物体的质量为 $m$ 以及其位置矢量为 $\vec r$
- 则垂直与位置矢量的分量为 $F\sin\theta$
- 根据力矩的定义，我们可以得出 $\tau = r F \sin \theta$，其中 $r$ 是位置矢量的模长
- 则根据上面我们推出的的转动惯量和力矩之间的关系，我们可以得到结果 $\vec r F\sin\theta = mr^2\vec \alpha$

---
## 刚体的转动惯量

- 连续体：$I=\int r^2dm$
- 复习一下质量元
  - 线密度 $dm = \lambda dl$
  - 面密度 $dm = \sigma dA$
  - 体密度 $dm = \rho dV$

![image](PIC/Chapter9-3.png)

对于上面这个例子，偶们来分析其转动惯量

- $I=\int r^2dm$
- $I=\int r^2 \lambda dl$
- $I=\int_{-L/2}^{L/2} x^2 {M\over L} dx$
- $I={1\over 12}ML^2$

![image](PIC/Chapter9-4.png)

继续分析这个新的例子

- $I=\int r^2dm$
- $I=\int r^2\lambda dl$
- $I=\int_0^L x^2 {M\over L}dx%$
- $I={1\over 3}x^3{M\over L}|_0^L$
- $I={1\over 3}{ML^2}$

![image](PIC/Chapter9-5.png)

继续分析这个新的例子

- $I=\int r^2dm$
- $I=\int r^2\lambda dl$
- $I=\int_0^{2\pi} R^2 {M\over 2\pi R} Rd\theta$
- $I={MR^2\over 2\pi} \int_o^{2\pi} d\theta$
- $I=MR^2$

![image](PIC/Chapter9-6.png)

继续分析这个新的例子

- 对于每一个半径为 $r$, 宽度为 $dr$ 的环来说，其 $I=r^2{2\pi rdr\over \pi R^2}M={2r^3M \over R^2}dr$
- 所以根据上述将0-R大小的环全部积分 $I=\int_0^R {2r^3M \over R^2}dr={1\over 2}MR^2$

![image](PIC/Chapter9-7.png)

继续分析这个新的例子

- 对于这样一个圆壳，我们需要在外侧取圆环求解，对于一个角度为 $\theta$ 的圆环 $I=\int r^2 dm = \int R^2\sin^2\theta \sigma ds$
- 现在来求解上面的 $ds=2\pi R\sin\theta d\theta$
- 所以对于这样一个圆环 $I=\int R^2\sin^2\theta {M\over 4\pi R^2} 2\pi R\sin\theta d\theta=\int {\sin^3\theta MR\over 2}d\theta$
- 根据范围进行积分 $I=\int_0^\pi {\sin^3\theta MR\over 2}d\theta={2\over 3}MR^2$

![image](PIC/Chapter9-8.png)

继续分析这个新的例子

- 对于一个半径为r的球壳，其 $I={2\over 3}\int r^2dm={2\over 3}\int r^2\rho dV$
- $dV=4\pi r^2 dr$
- 所以 $I={2\over 3}\int_0^R r^2 4\pi r^2 {M\over {4\over 3}\pi R^3}dr={2\over 3}\int_0^R {3r^4M\over R^3}dr={2\over 5}MR^2$

![image](PIC/Chapter9-9.png)

相同的方法横向切片视为多个圆盘的叠加也可以完成计算，这里就不赘述了

## 本节总结

| 物体/模型 | 取法或微元 | 转动惯量积分表达 | 结果 |
| --- | --- | --- | --- |
| 细杆，转轴过中点且垂直杆 | $dm=\lambda dl$，$x\in[-L/2,L/2]$ | $I=\int_{-L/2}^{L/2} x^2 {M\over L}dx$ | $I={1\over 12}ML^2$ |
| 细杆，转轴过一端且垂直杆 | $dm=\lambda dl$，$x\in[0,L]$ | $I=\int_0^L x^2 {M\over L}dx$ | $I={1\over 3}ML^2$ |
| 细圆环 | $dm=\lambda dl$，$r=R$ | $I=\int_0^{2\pi} R^2 {M\over 2\pi R}R d\theta$ | $I=MR^2$ |
| 圆盘 | 取半径为 $r$、宽度为 $dr$ 的微元圆环 | $dI= r^2 {2\pi r dr\over \pi R^2}M$，$I=\int_0^R {2r^3M\over R^2}dr$ | $I={1\over 2}MR^2$ |
| 圆壳 | 取角度为 $\theta$ 的环带，$r=R\sin\theta$ | $I=\int_0^\pi R^2\sin^2\theta {M\over 4\pi R^2}2\pi R\sin\theta d\theta$ | $I={2\over 3}MR^2$ |
| 实心球 | 视为薄球壳叠加，$dV=4\pi r^2dr$ | $I={2\over 3}\int_0^R r^2 4\pi r^2 {M\over {4\over 3}\pi R^3}dr$ | $I={2\over 5}MR^2$ |

这些例子的核心思路都是先选取合适的质量元，再把 $r^2dm$ 写成单变量积分，最后按几何范围积分即可。

## 牛顿定律在转动方面的应用

![image](PIC/Chapter9-10.png)

对于上面这个例子我们进行如下分析

首先题目中给出了必要的信息：重量为 $m$，半径为 $R$ 的定滑轮，以及一左一右两个质量为 $m_1$, $m_2$. 我们假设 $m_1>m_2$ 来对题目进行分析
- $m_1-T_1=m_1a$
- $T_2-m_2=m_2a$

根据上面两个式子我们可以得到
- $T_1={2m_2+m/2\over m_1+m_2+m/2}m_1g$
- $T_2={2m_1+m/2\over m_1+m_2+m/2}m_2g$

根据上述式子重新带入原有式子可以得到

$a={m_1-m_2\over m_1+m_2+m/2}g$
则定滑轮的角加速度为 $\vec \alpha = {a/over R}= {m_1-m_2\over m_1+m_2+m/2}{g\over R}$

![image](PIC/Chapter9-11.png)

有了上面的计算基础之后，我们继续分析这个稍微复杂一些的例子

- $m_1g-T_1=m_1a_1$
- $T_2-m_2g=m_2a_2$
- $T_1R-T_2r=Ia$

根据两个滑轮的大小和半径，我们可以先求得他们的转动惯量大小

- $I={1\over 2}M_1R^2+{1\over 2}M_1=1r^2=0.035\ kg\cdot m^2$

然后我们要推断角加速度和加速度的直接关联

- $a_1=R\alpha=0.1\alpha$
- $a_2=R\alpha=0.05\alpha$

将上述两个式子带入到最开始的式子中我们可以得到

- $T_1=2g-0.2\alpha$
- $T_2=2g+0.1\alpha$

将上述两个式子带入到转动惯量的等式之中

$(2g-0.2\alpha)\cdot 0.1- (2g+0.1\alpha)\cdot 0.05 =0.035\alpha$
$\alpha = 16.33\ rad/s^2$ 

再根据其值求解其他所有的结果