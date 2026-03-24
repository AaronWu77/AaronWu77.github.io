# Chapter 2 A Quantitive Approach

## What is pipelining?

- 一种实现技术，通过同时重叠执行不同的指令来实现。
- 一种用于制造快速 CPU 的实现技术。

![alt text](PIC/PIC3-1.png)

**Conclusion**
- 实现快速 CPU 的关键实施技术在于：减少 CPU 时间。decrease CPUtime.
- 提高吞吐量（而非单个执行时间） Improving of Throughput ( rather than individual execution time)
- 提高资源（功能单元）的效率 Improving of efficiency for resources  (functional unit)

**Ideal Performance for Pipelining**

If the stages are perfectly balanced, The time per instruction on the pipelined processor equal to:
$$\frac{ A} {B C}$$
