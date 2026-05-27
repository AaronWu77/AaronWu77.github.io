### **4**

**证明**  
记 \(n = \dim V\)。已知 \(\operatorname{null} T^{n-2} \ne \operatorname{null} T^{n-1}\)，这意味着 \(T^{n-2}\) 的零空间在 \(T^{n-1}\) 时严格变大。  
考虑 \(T\) 的 Jordan 标准形（因 \(V\) 是复向量空间）。设 \(J\) 的最大 Jordan 块大小为 \(k\)。  
- 若 \(k \le n-2\)，则 \(T^{n-2} = 0\) 在最大块上，从而 \(\operatorname{null} T^{n-2} = \operatorname{null} T^{n-1}\)，矛盾。  
- 故 \(k \ge n-1\)。  
最大块大小 \(k\) 只能为 \(n-1\) 或 \(n\)。  
- 若 \(k = n\)，则只有一个特征值。  
- 若 \(k = n-1\)，则另一个特征值对应一个 1 维 Jordan 块，因此恰有两个特征值。  
不可能有三个及以上特征值，否则所有块大小 ≤ \(n-2\)，矛盾。  
因此 \(T\) 最多有两个不同的特征值。

---

### **10**

**证明**  
由 Jordan 标准形定理，存在可逆矩阵 \(P\) 使得  
\[
P^{-1} T P = J = \operatorname{diag}(J_1, J_2, \dots, J_m),
\]  
其中每个 \(J_i\) 是 Jordan 块：\(J_i = \lambda_i I_{r_i} + M_i\)，\(M_i\) 是幂零的（严格上三角）。  
定义  
\[
D' = \operatorname{diag}(\lambda_1 I_{r_1}, \lambda_2 I_{r_2}, \dots, \lambda_m I_{r_m}), \quad N' = J - D'.
\]  
则 \(D'\) 是对角矩阵，\(N'\) 是幂零矩阵，且 \(D' N' = N' D'\)（对角与块对角可交换）。  
令  
\[
D = P D' P^{-1}, \quad N = P N' P^{-1}.
\]  
那么 \(T = D + N\)。  
- \(D\) 可对角化（相似于对角矩阵）。  
- \(N\) 幂零（相似于幂零矩阵）。  
- \(DN = P D' P^{-1} P N' P^{-1} = P D' N' P^{-1} = P N' D' P^{-1} = N D\)。  
证毕。

---

### **8**

**证明**  
设 \(n = \dim V\)。  
“\(\Leftarrow\)”：极小多项式 \(m_T(z) = (z - \lambda)^n\)。此时 \(T\) 的 Jordan 形是单个 \(n \times n\) Jordan 块。  
若存在两个非平凡 \(T\)-不变子空间 \(U, W\) 使 \(V = U \oplus W\)，则 \(T\) 在 \(U\) 和 \(W\) 上的限制对应的 Jordan 块数至少为 2，这与 Jordan 形是单块矛盾。  
故 \(V\) 不能分解为两个 \(T\)-不变真子空间的直和。  

“\(\Rightarrow\)”：假设 \(V\) 不能分解为两个 \(T\)-不变真子空间的直和。  
- 若 \(T\) 有两个不同的特征值 \(\lambda, \mu\)，则广义特征空间 \(G_\lambda, G_\mu\) 都是非平凡 \(T\)-不变子空间，且 \(V = G_\lambda \oplus G_\mu\)，矛盾。  
- 故 \(T\) 只有一个特征值 \(\lambda\)。于是 \(T - \lambda I\) 是幂零算子。若 \((T - \lambda I)^{n-1} \ne 0\)，则极小多项式次数为 \(n\)，即 \((z - \lambda)^n\)。  
若 \((T - \lambda I)^{n-1} = 0\)，则次数 ≤ \(n-1\)，此时 \(T\) 的有理标准形至少有两个循环子空间，从而可分解为直和，矛盾。  
因此极小多项式为 \((z - \lambda)^n\)。

---

### **6**

**证明**  
由 Jordan 基理论，存在向量 \(v_1, \dots, v_n\) 和非负整数 \(m_1, \dots, m_n\) 使得  
\[
\{ N^j v_i \mid 1 \le i \le n, \; 0 \le j \le m_i \}
\]  
构成 \(V\) 的一组基，且 \(N^{m_i+1} v_i = 0\)，\(N^{m_i} v_i \ne 0\)。  
令 \(w_i = N^{m_i} v_i\)，则 \(N w_i = 0\)，故 \(w_i \in \operatorname{null} N\)。  
若 \(\sum_{i=1}^n a_i w_i = 0\)，对每个 \(k\) 作用 \(N^{m_k}\) 可得 \(a_k = 0\)（因为 \(N^{m_k} w_i \ne 0\) 仅当 \(i = k\) 且 \(m_k \ge m_i\) 可严格论证）。  
因此 \(w_1, \dots, w_n\) 线性无关。  
由于 \(\dim \operatorname{null} N = n\)（等于 Jordan 链的条数），它们构成 \(\operatorname{null} N\) 的一组基。  
特别地，\(n = \dim \operatorname{null} N\) 与 Jordan 基选取无关。

---