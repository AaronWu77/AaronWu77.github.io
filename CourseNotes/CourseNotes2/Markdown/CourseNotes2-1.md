# Chapter 1 Measurement

## The International System of Units -- SI
<!-- Standard Academic Table ("Three-Line Table") Style in HTML -->
<table style="width: 100%; border-collapse: collapse; border-top: 2px solid #e0e0e0; border-bottom: 2px solid #e0e0e0; text-align: left; background-color: transparent;">
    <thead>
        <tr style="border-bottom: 1px solid #e0e0e0;">
            <th style="padding: 10px; font-weight: bold;">Quantity</th>
            <th style="padding: 10px; font-weight: bold;">Name</th>
            <th style="padding: 10px; font-weight: bold;">SI Unit</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td style="padding: 8px;">Time</td>
            <td style="padding: 8px;">second</td>
            <td style="padding: 8px;"><code>s</code></td>
        </tr>
        <tr>
            <td style="padding: 8px;">Length</td>
            <td style="padding: 8px;">meter</td>
            <td style="padding: 8px;"><code>m</code></td>
        </tr>
        <tr>
            <td style="padding: 8px;">Mass</td>
            <td style="padding: 8px;">kilogram</td>
            <td style="padding: 8px;"><code>kg</code></td>
        </tr>
        <tr>
            <td style="padding: 8px;">Amount of substance</td>
            <td style="padding: 8px;">mole</td>
            <td style="padding: 8px;"><code>mole</code></td>
        </tr>
        <tr>
            <td style="padding: 8px;">Thermodynamic</td>
            <td style="padding: 8px;">kelvin</td>
            <td style="padding: 8px;"><code>K</code></td>
        </tr>
        <tr>
            <td style="padding: 8px;">Electric current</td>
            <td style="padding: 8px;">ampere</td>
            <td style="padding: 8px;"><code>A</code></td>
        </tr>
        <tr>
            <td style="padding: 8px;">Luminious intensity</td>
            <td style="padding: 8px;">candela</td>
            <td style="padding: 8px;"><code>cd</code></td>
        </tr>
    </tbody>
</table>

## Dimension and dimensional analysis

对于一个等式，或者是计算的结果，可以先通过对其单位的判断进行简单的检查，比如：
- $[x] = L$
- $[v] = LT^{-1}$
- $[a] = LT^{-2}$
- $[f] = MLT^{-2}$

# Chapter 2 Motion in 1-D (Kinematics)


## Properties of Vector

**向量 (vector)**：一种包含大小 (magnitude) 和 方向 (direction) 的表示方式

**向量的分解**：可以根据直角坐标系分解为单位向量

![alt text medium](PIC/PIC1-1.png)
![alt text medium](PIC/PIC1-2.png)

**向量的加法**：三角形/四边形法则

**向量的乘法**：
- $a\vec B = \vec C$
- $\vec A \cdot \vec B = AB cos \theta=c$
- $\vec A \times \vec B = \vec C$

![alt text medium](PIC/PIC1-3.png)

从单位向量的角度看向量的乘法：
- $\vec a = a_x \hat i + a_y \hat j + a_z \hat k$
- $\vec b = b_x \hat i + b_y \hat j + b_z \hat k$
- $\vec a \cdot \vec b = a_x b_x + a_y b_y + a_z b_z$
- $\vec a \times \vec b = (a_y b_z - a_z b_y) \hat i + (a_z b_x - a_x b_z) \hat j + (a_x b_y - a_y b_x) \hat k$

点乘和叉乘的一些交换：
- $\vec a \cdot \vec b = \vec b \cdot \vec a$
- $\vec a \times \vec b = -\vec b \times \vec a$
- $\vec a \cdot (\vec b \times \vec c) = \vec c \cdot (\vec a \times \vec b) = \vec b \cdot (\vec c \times \vec a)$
- $\vec a \times (\vec b \times \vec c) = \vec b (\vec a \cdot \vec c) - \vec c (\vec a \cdot \vec b)$



