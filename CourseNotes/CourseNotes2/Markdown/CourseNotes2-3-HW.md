---
title: Homework 1
author: 吴宇宸
date: 2026-03-09
---

# <center>Homework 1</center>

**姓名：** 吴宇宸  
**学号：** 3230100789 
**日期：** 2026.3.9

---

## Ex.1 (P63,T7)
A child's toy consists of three cars that are pulled in tandem (串联工作组) on small frictionless rollers (Fig. 3-33). The cars have masses $m_1 = 3.1 kg$, $m_2 = 2.4 kg$, and $m_3 = 1.2 kg$. If they are pulled to the right with a horizontal force  $P = 6.5 N$, find
(a) the acceleration of the system
(b) the force exerted by thesecond car on the third car
(c) the force exerted by thefirst car on the second car.
![alt text](./PIC/PIC3-HW-1.png)

### Answer

**(a)** To analyse the whole system, we only need to consider the horizontal force $P$ since the rollers are frictionless.
Thus, 
$$
\begin{align}(m_1+m_2+m_3)\cdot a &= P\\
a &= \frac{P}{m_1+m_2+m_3} \\
a &= \frac{6.5 N}{3.1\ kg +2.4\ kg + 1.2\ kg}\\
a &\approx 0.97\ m/s^2
\end{align}
$$
The direction of $a$ is right.

**(b)** Looking at the third car, the force exerted by the second car on the third car causes the acceleration of the third car. 
Thus:
$$
\begin{align}F_{23} &= m_3\cdot a\\
F_{23} &= 1.2\ kg \cdot 0.97\ m/s^2 \\
F_{23} &\approx 1.16\ N \\
\end{align}
$$
The direction of $F_{23}$ is right.

**(c)** We perceive the second and third car as a tandem, thus the force exerted by the first car on the second car causes the acceleration of the tandem.
Thus:
$$
\begin{align}F_{12} &= {(m_3+m_2)}\cdot a\\
F_{12} &= (1.2\ kg+2.4\ kg) \cdot 0.97\ m/s^2 \\
F_{12} &\approx 3.49\ N \\
\end{align}
$$
The direction of $F_{12}$ is right.

---

## Ex.2 (P63,T9)

A chain consisting of five links, each with mass $100 g$, is lifted vertically with a constant acceleration of $2.50 m/s^2$, as shown in Fig. 3-35. Find 
(a) the forces acting between adjacent links
(b) the force F exerted on the top link by the agent lifting the chain
(c) the net force (合力) on each link
![alt text](./PIC/PIC3-HW-3.png)

### Answer
**(a)** Assume the force exerted by the top adjacent to the second as $F_{12}$, it is easy to find that:

$$
\begin{align}
F_{12} &= 4\cdot m \cdot (a+g)\\
F_{23} &= 3\cdot m \cdot (a+g) \\
F_{34} &= 2\cdot m \cdot (a+g) \\
F_{45} &= 1\cdot m \cdot (a+g) \\
\end{align}
$$

Thus: $ F_{12} = 5\ N; F_{23} = 3.75\ N; F_{34} = 2.5\ N; F_{45} = 1.25\ N;$

**(b)** The force F exerted on the top link by the agent lifting the chain cause the acceleration of the whole chain. Thus,
$$
\begin{align}
F &= 5\cdot m \cdot (a+g)\\
F &= 5\cdot 0.1\ kg\cdot (2.50\ m/s^2+10.0\ m/s^2) \\
F &= 6.25\ N\\
\end{align}
$$
**(c)** On each link, the net force means the acceleration plus the mass of the link.
$F_{1} = F_{2} =F_{3} =F_{4} =F_{5} = m \cdot a = 0.25\ N$

---

## Ex.3 (P86,T15)

A body of mass $m$ falls from rest through the air. A drag force $D = bv^2$ opposes the motion of the body. 
(a) What is the initial downward acceleration of the body? 
(b) After some time the speed of the body approaches a constant value. What is this terminal speed $v_T$?
(c) What is the downward acceleration of the body when $v = v_T/2$?

### Answer
**(a)** When t=0, the body's velocity equals to 0 which means the drag force is also 0. Thus the initial downward acceleration of the body is $g$.

**(b)** When the body reaches a constant value, the gravity equals to the drag force, which means:
$$\begin{align} 
m\cdot g &= bv^2\\
v&=\sqrt{\frac{m\cdot g}{b}}
\end{align}
$$

**(c)** When $v = v_T/2$, we can know that:
$$ \begin{align}
m\cdot a &= m\cdot g- b(\frac{\sqrt{\frac{m\cdot g}{b}}}{2})^2\\
a &=g-\frac{b\cdot m\cdot g}{b\cdot 4\cdot m}\\
a &=\frac{3}{4} g\\
\end{align}
$$

---

## Ex.4 (P86,T18)

(a) Assuming that the drag force $D$ is given by $D = bv$, show that the distance $y_{95}$ through which an object must fall from rest to reach 95% of its terminal speed is given by
$$y_{95} = \frac{v_T^2}{g} (ln(20)-\frac{19}{20})$$
where $v_T$ is the terminal speed. (Hint: Use the result for $y(t)$ ob-tained in Problem 17.) 
(b) Using the terminal speed of $42 m/s$ for the baseball given in Table 4-1, calculate the 95% distance.Why does the result not agree with the value listed in Table 4-1?
![alt text](./PIC/PIC3-HW-4.png)

### Answer
**(a)** Using the result for $v(t)$ and $y(t)$ obtained in Problem 17:
$$ v(t) = v_T \left(1 - e^{-\frac{gt}{v_T}}\right) $$
$$ y(t) = v_T t - \frac{v_T^2}{g} \left(1 - e^{-\frac{gt}{v_T}}\right) $$
We are looking for the distance $y_{95}$ when the object reaches 95% of its terminal speed, meaning $v(t_{95}) = 0.95 v_T$. 
Substitute this into the velocity equation:
$$ 0.95 v_T = v_T \left(1 - e^{-\frac{g t_{95}}{v_T}}\right) \implies e^{-\frac{g t_{95}}{v_T}} = 0.05 = \frac{1}{20} $$
Taking the natural logarithm, we find the time $t_{95}$:
$$ t_{95} = \frac{v_T}{g} \ln(20) $$
Now, substitute $t_{95}$ and $e^{-\frac{gt_{95}}{v_T}} = \frac{1}{20}$ back into the $y(t)$ equation:
$$ 
\begin{align}
y_{95} &= v_T \left( \frac{v_T}{g} \ln(20) \right) - \frac{v_T^2}{g} \left( 1 - \frac{1}{20} \right) \\
y_{95} &= \frac{v_T^2}{g} \left( \ln(20) - \frac{19}{20} \right)
\end{align}
$$

**(b)** Using $v_T = 42\ m/s$ and $g = 9.8\ m/s^2$:
$$ 
y_{95} = \frac{(42)^2}{9.8} \left(\ln(20) - 0.95\right) \approx 180 \times (2.996 - 0.95) \approx 368\ m 
$$
**Why it does not agree:** The result is quite different from empirical table values because a fast-moving baseball experiences aerodynamic drag proportional to $v^2$ ($D = cv^2$), rather than the simple linear viscous drag ($D = bv$) assumed in this problem.

---

## Ex.5 (P86,T20)

A particle $P$ travels with constant speed on a circle of radius $3.0 m$ and completes one revolution in 20 s (Fig. 4-46). Theparticle passes through $O$ at $t = 0$. With respect to the origin $O$, find 
(a) the magnitude and direction of the vectors describing its position 5.0, 7.5, and 10 s later
(b) the magnitude anddirection of the displacement in the 5.0-s interval from thefifth to the tenth second, 
(c) the average velocity vector in this interval
(d) the instantaneous velocity vector at the beginning and at the end of this interval
(e) the instantaneous accel-eration vector at the beginning and at the end of this interval.Measure angles counterc lockwise from the $x$ axis.
![alt text](./PIC/PIC3-HW-6.png)

### Answer
The period is $T = 20\ s$, meaning the angular speed is $\omega = \frac{2\pi}{T} = \frac{\pi}{10}\ rad/s$, and the constant linear speed is $v = \omega R = 3.0(\frac{\pi}{10}) \approx 0.94\ m/s$. The trajectory is a circle of radius $R = 3.0\ m$ passing through the origin. We put its center at $(3.0, 0)$ so it starts at $O(0,0)$ and moves counter-clockwise.

**(a)** Position vectors $\vec{r}$:
*   **$t = 5.0\ s$**: Completes $1/4$ circle ($90^\circ$). Position is $(3.0, 3.0)\ m$. 
    Magnitude: $\sqrt{3.0^2 + 3.0^2} \approx 4.2\ m$. Direction: $45^\circ$.
*   **$t = 7.5\ s$**: Completes $3/8$ circle ($135^\circ$). By checking the geometry from the center, the coordinates are $(3.0 - 3.0\cos 45^\circ, 3.0\sin 45^\circ) \approx (0.88, 2.12)\ m$. 
    Magnitude: $\approx 2.3\ m$. Direction: $\approx 67.5^\circ$. *(Note: using coordinates from origin)*
*   **$t = 10\ s$**: Completes $1/2$ circle ($180^\circ$). Position is $(6.0, 0)\ m$. 
    Magnitude: $6.0\ m$. Direction: $0^\circ$.

**(b)** Displacement ($\Delta\vec{r}$) from $t=5.0\ s$ to $10\ s$:
$$ \Delta\vec{r} = \vec{r}_{10} - \vec{r}_5 = (6.0\hat{i}) - (3.0\hat{i} + 3.0\hat{j}) = (3.0\hat{i} - 3.0\hat{j})\ m $$
Magnitude: $\sqrt{3^2 + (-3)^2} \approx 4.2\ m$. Direction: $-45^\circ$ (or $315^\circ$).

**(c)** Average velocity $\vec{v}_{avg}$:
$$ \vec{v}_{avg} = \frac{\Delta\vec{r}}{\Delta t} = \frac{3.0\hat{i} - 3.0\hat{j}}{5.0\ s} = (0.60\hat{i} - 0.60\hat{j})\ m/s $$
Magnitude: $\sqrt{0.6^2 + (-0.6)^2} \approx 0.85\ m/s$. Direction: $-45^\circ$.

**(d)** Instantaneous velocity $\vec{v}$: The magnitude is constant $v \approx 0.94\ m/s$, directed tangent to the path.
*   **$t=5.0\ s$**: At the top of the circle, moving exactly matching the x-axis. Direction: $0^\circ$.
*   **$t=10\ s$**: At the far right of the circle, moving exactly downwards. Direction: $-90^\circ$ (or $270^\circ$).

**(e)** Instantaneous acceleration $\vec{a}$: The magnitude is constant $a = \frac{v^2}{R} = R\omega^2 \approx 0.30\ m/s^2$, always pointing to the center of the circle $(3.0, 0)$.
*   **$t=5.0\ s$**: At $(3.0, 3.0)$, pointing straight down to the center. Direction: $-90^\circ$ (or $270^\circ$).
*   **$t=10\ s$**: At $(6.0, 0)$, pointing straight left to the center. Direction: $180^\circ$.

---

## Ex.6 (P87,T23)

A particle moves in a plane according to
$x=R\sin \omega t +\omega Rt$
$y=R\cos \omega t +R$,
where $\omega$ and $R$ are constants. This curve, called a cycloid, is the path traced out by a point on the rim of a wheel that rolls without slipping along the x axis. 
(a) Sketch the path. 
(b) Calculate the instantaneous velocity and acceleration when the particle is at its maximum and minimum value of y.

### Answer
**(b)** Before finding the specific conditions, let's first calculate the velocity components ($v_x, v_y$) and acceleration components ($a_x, a_y$) by taking the first and second time derivatives of $x$ and $y$:

Velocity:
$$
\begin{align}
v_x &= \frac{dx}{dt} = R\omega \cos \omega t + \omega R \\
v_y &= \frac{dy}{dt} = -R\omega \sin \omega t
\end{align}
$$

Acceleration:
$$
\begin{align}
a_x &= \frac{dv_x}{dt} = -R\omega^2 \sin \omega t \\
a_y &= \frac{dv_y}{dt} = -R\omega^2 \cos \omega t
\end{align}
$$

We separate this problem into two sub-problems: 
1. When the particle is at its maximum value of $y$.
2. When the particle is at its minimum value of $y$.

**1. Maximum value of $y$**
The maximum value of $y$ occurs when $\cos \omega t = 1$ (e.g., at $\omega t = 0$). At this moment, $y = 2R$, $\sin \omega t = 0$.
Substituting these trigonometric values:
*   **Velocity:** $v_x = R\omega(1) + \omega R = 2R\omega$, and $v_y = 0$. 
    Thus, the instantaneous velocity is $2R\omega$ directed along the positive x-axis.
*   **Acceleration:** $a_x = -R\omega^2(0) = 0$, and $a_y = -R\omega^2(1) = -R\omega^2$.
    Thus, the instantaneous acceleration is $R\omega^2$ directed along the negative y-axis (downward).

**2. Minimum value of $y$**
The minimum value of $y$ occurs when $\cos \omega t = -1$ (e.g., at $\omega t = \pi$). At this moment, $y = 0$, $\sin \omega t = 0$.
Substituting these trigonometric values:
*   **Velocity:** $v_x = R\omega(-1) + \omega R = 0$, and $v_y = 0$. 
    Thus, the instantaneous velocity is $0$.
*   **Acceleration:** $a_x = -R\omega^2(0) = 0$, and $a_y = -R\omega^2(-1) = R\omega^2$.
    Thus, the instantaneous acceleration is $R\omega^2$ directed along the positive y-axis (upward).

---