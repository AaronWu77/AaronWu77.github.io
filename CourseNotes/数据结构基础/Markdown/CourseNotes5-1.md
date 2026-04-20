# Chapter 2 Algorithm Analysis

## Definition

An algorithm is a finite set of instructions that, if allowed, accomplishes a particular task. In addition, all algorithms must satisfy the following criteria:

- **Input**: There are *zero or more* quantities that are externally supplied
- **Output**: At least one quantity is produced
- **Definiteness**: Each instruction is *clear and unambiguous*
- **Finiteness**: The algorithm must terminate after a *finite* number of steps
- **Effectiveness**: Every instruction must be basic enough to be carried out, in principle, by a person using only pencil and paper.

> 总结：一个算法包含0个或多个输入，一个及以上的输出，确定性，有限性，有效性

**Program and Algorithm**
- A *Program* is writeen in some programming language, and does not have to be finite.(Operation system, web server, etc.)
- An *Algorithm* can be described by human languages, flow charts, some programming languages, or pseudocode

> *程序*是用某种编程语言编写的，并且不一定具有有限性（例如操作系统）。
> *算法*可以用人类语言、流程图、某些编程语言或伪代码来描述。

## What to Analyze

**Time and Space Comlexity**
- $T_{avg}(N)$: Average time complexity (with the input size of N)
- $T_{worst}(N)$: Worst-case time complexity (with the input size of N)

**Example**

![alt text](PIC/PIC1-1.png)

## Asymptotic Notation

- $T(N) = O(f(N))$: There exist positive constants $c$ and $N_0$ such that $T(N) \leq c \cdot f(N)$ for all $N \geq N_0$.
- $T(N) = \Omega(f(N))$: There exist positive constants $c$ and $N_0$ such that $T(N) \geq c \cdot f(N)$ for all $N \geq N_0$.
- $T(N) = \Theta(f(N))$: There exist positive constants $c_1$, $c_2$, and $N_0$ such that $c_1 \cdot f(N) \leq T(N) \leq c_2 \cdot f(N)$ for all $N \geq N_0$.
- $T(N) = o(f(N))$: For any positive constant $c$, there exists a constant $N_0$ such that $T(N) < c \cdot f(N)$ for all $N \geq N_0$.

> * $T(N) = O(f(N))$: 存在正的常数 $c$ 和 $N_0$，使得对于所有 $N \geq N_0$，都有 $T(N) \leq c \cdot f(N)$。
> * $T(N) = \Omega(f(N))$: 存在正的常数 $c$ 和 $N_0$，使得对于所有 $N \geq N_0$，都有 $T(N) \geq c \cdot f(N)$。
> * $T(N) = \Theta(f(N))$: 存在正的常数 $c_1$、$c_2$ 和 $N_0$，使得对于所有 $N \geq N_0$，都有 $c_1 \cdot f(N) \leq T(N) \leq c_2 \cdot f(N)$。
> * $T(N) = o(f(N))$: 对于任何正的常数 $c$，存在一个常数 $N_0$，使得对于所有 $N \geq N_0$，都有 $T(N) < c \cdot f(N)$。

**Example:**

- $2N + 3 = O(N) = O(N^2) ....$, we shall always take the smallest one, so $2N + 3 = O(N)$
- $2^N + N^2 = \Omega(2^N)=\Omega(N)$, we shall always take the largest one, so $2N + 3 = \Omega(N)$

**Rules of Asymptotic Notation**

- If $T_1(N) = O(f(N))$ and $T_2(N) = O(g(N))$ then
    - $T_1(N) + T_2(N) = O(max(f(N), g(N)))$
    - $T_1(N) \cdot T_2(N) = O(f(N) \cdot g(N))$
- If $T(N)$ is a polynomial of degree $k$ then $T(N) = O(N^k)$
- $log^k N = O(N)$ for any constant $k$.
