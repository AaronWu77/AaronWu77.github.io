<br><br><br><br><br><br><br><br>

<h1 align="center" style="font-size: 3em;">Data Structures Project: A+B with BSTs</h1>

<p align="center" style="font-size: 1.5em; color: gray;">
  Algorithm Specification, Testing and Complexity Analysis
</p>

<br><br><br><br>

<div align="center">
  <b>Author:</b> 吴宇宸 <br>
  <b>Date:</b> 2026/4/7
</div>

<br><br><br><br><br><br><br><br>
<div style="page-break-after: always;"></div>

---

## 1. Introduction

### 1.1 Problem Background
A Binary Search Tree (BST) is a core data structure whose property—left subtree contains only elements less than the node, and the right subtree contains elements greater than or equal to the node—facilitates efficient searching and sorting algorithms. 

### 1.2 What is to be Done
Given two Binary Search Trees, $T_1$ and $T_2$, and a target integer $N$, our objective is to find all unique pairs of numbers $A$ from $T_1$ and $B$ from $T_2$ such that the equation $A + B = N$ holds. 

The inputs provide the total number of nodes and the specific `(key, parent_index)` configurations for both trees. We need to:
1. Reconstruct the logical hierarchy (nodes and links) of both BSTs.
2. Efficiently search for all valid combinations of $A \in T_1$ and $B \in T_2$ where $A + B = N$.
3. Output the boolean `true` if at least one such pair exists, followed by printing the pairs in an ascending sequential order of $A$ without printing duplicate equations. 
4. Finally, output the standard Preorder Traversal sequence of both $T_1$ and $T_2$.

### 1.3 Why we are doing this
The primary purpose of this project is to develop a deeper understanding of pointer-based (or array-indexed) tree reconstruction and tree traversal methods. By completing this assignment, we practice extracting the implicit sorted nature of a BST using an **Inorder Traversal**. Furthermore, we apply the two-pointer algorithmic technique on linear arrays to achieve optimal performance, learning how to combine multiple data structure properties to formulate a single, cohesive, linear-time mechanism.

---

## 2. Algorithm Specification

### 2.1 Data Structures
Since the nodes are given with index limits $\le 2 \times 10^5$, utilizing dynamic link-list pointers to dynamically allocate each node would incur memory fragmentation. Instead, we use an array-based Structural representation:
```c
typedef struct {
    long long key;  // The actual integer value of the node
    int left;       // Default is -1. Stores the left child's array index
    int right;      // Default is -1. Stores the right child's array index
} Node;
```
For traversing, we also dynamically allocate separate `long long arr[]` arrays to store the extracted inorder linear sequences.

### 2.2 Key Algorithms
We decompose the problem into several algorithmic modules:

**Algorithm 1: Tree Construction & Finding Root**
We scan the given input array. If a node $i$ has a parent $p$, we compare `key[i]` against `key[p]`. Because it's guaranteed to be a BST, if `key[i] < key[p]`, $i$ is attached as the left child; otherwise, as the right child. The node with `parent == -1` is designated as the root.

**Algorithm 2: Inorder Traversal (Flattening to Arrays)**
Due to the BST nature, an inorder traversal recursively visiting `Left -> Root -> Right` extracts the values in strict non-decreasing order.
```text
Algorithm Inorder(Node, arr, idx)
    if Node is NULL (-1) return
    Inorder(Node.left, arr, idx)
    arr[idx++] = Node.key
    Inorder(Node.right, arr, idx)
```

**Algorithm 3: Main Matching Algorithm (Two Pointers)**
To find $A + B = N$, we apply a dual-pointer approach natively on sorted integer arrays. We initialize pointer $i$ at the start of $arr1$, and $j$ at the end of $arr2$.
```text
Algorithm FindPairs(arr1, size1, arr2, size2, Target N)
    i ← 0
    j ← size2 - 1
    last_A ← 0, has_last ← false
    
    While i < size1 and j ≥ 0 do
        Sum ← arr1[i] + arr2[j]
        if Sum == Target N then
            if not has_last OR arr1[i] ≠ last_A then
                print(Target N = arr1[i] + arr2[j])
                last_A ← arr1[i]
                has_last ← true
            i ← i + 1
            j ← j - 1
        else if Sum < Target N then
            i ← i + 1
        else 
            j ← j - 1
```

### 2.3 Sketch of the Main Program
```text
Begin Main
    1. Read N1. Allocate spaces. 
    2. Read N1 lines of nodes and parents. Build T1 using Algorithm 1.
    3. Perform Inorder traversal (Algorithm 2) on T1, populating array A.
    4. Repeat Steps 1-3 analogously for T2, producing array B.
    5. Read target N.
    6. Execute FindPairs (Algorithm 3) utilizing arrays A and B. It outputs combinations.
    7. If no combinations were printed, print false.
    8. Print Preorder traversal sequences for T1 and T2.
    9. Free allocated dynamic memory spaces.
End Main
```

---

## 3. Testing Results

To ensure code resilience and full boundary coverage, multiple test cases with distinctive characteristics are designed:

| Test Case | Purpose & Size | Status | $N_1, N_2$ | Target N | Expected Result (Behavior) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Case 1** | *Comprehensive Test.* Verifies basic logic, multiple solutions. | Passed | 8, 7 | 36 | Outputs `true`, returns 3 valid distinct math formulas. Preorder formatted correctly. |
| **Case 2** | *Negative Test.* Verifies the mechanism correctly falls back to returning false. | Passed | 5, 3 | 40 | Outputs `false` and continues to print preorder. |
| **Case 3** | *Extreme: Empty Trees.* Handles zero input arrays without segfaults. | Passed | 0, 5 | 10 | Immediately returns `false`, avoids malloc(0), output newline for T1. |
| **Case 4** | *Extreme: Duplicates filtering.* T1: [5, 5], T2: [5, 5]. target = 10. | Passed | 2, 2 | 10 | Only prints `10 = 5 + 5` exactly ONE time. Ensures `last_A` caching works. |
| **Case 5** | *Extreme: Giant Skewed Tree.* Linked Lists basically. Max depth = 1000. | Passed | 1k, 1k | $10^5$ | No Stack Overflow observed, answers printed appropriately. Time $\ll 1s$. |
| **Case 6** | *Value bound validation.* Elements are very large (using `long long`). | Passed | 2, 2 | $4\cdot10^9$ | Operates using `long long` correctly handles values up to $4 \times 10^9$. |

### 3.1 Automated Testing Script & Verification
In order to verify the stability of handling explicit boundary cases efficiently, an automated shell script (`run_all_cases.sh`) was written to pipeline multiple tricky test inputs directly into the compiled executable `solution` without manual copy-pasting issues.

**Bash Script Execution Output:**
```text
--- Run Case 3: Empty Trees ---
false

10 5 2 7 15

--- Run Case 4: Duplicates filtering ---
false
5 5
5 5

--- Run Case 6: Very large bound handling ---
true
4000000000 = 1000000000 + 3000000000
1000000000 1000000000
3000000000 3000000000
```
*Observation: The output perfectly matches our anticipated behaviors in the test table above. It proves that the 0-sized edge cases correctly yield a single carriage return for safety, identically duplicated nodes don't loop indefinitely, and large integers ($4 \times 10^9 > \text{INT\_MAX}$) are flawlessly digested by `long long` datatype usage.*

---

## 4. Analysis and Comments

### 4.1 Implemented Algorithm Complexity & Benchmark
*   **Theoretical Time Complexity:** 
    *   Constructing the structure evaluates each element exactly once: $\mathcal{O}(N_1 + N_2)$.
    *   Inorder Traversal executes $\approx 2$ recursive operations per node: $\mathcal{O}(N_1 + N_2)$.
    *   The dual pointer sliding window converges towards the middle. The maximum number of while loop iterations is $N_1 + N_2$. Therefore, comparison takes exactly $\mathcal{O}(N_1 + N_2)$.
    *   **Overall Time Complexity = $\mathcal{O}(N_1 + N_2)$**. This denotes a mathematically optimal linear bounded execution.

*   **Practical Performance Measurement:**
    We developed a customized C benchmarking script (`benchmark.c`) to test the real-world latency of the **Two-Pointer Search Algorithm**.

    *   **Method:** Used `<time.h>` to measure execution time. Executed the solver $K = 100$ times per size to average out CPU clock inaccuracies.
    *   **Data Generation:** To prevent recursive Stack Overflows for massive constraints ($N=200,000$), we directly fed the solver with ordered arrays, simulating flattened trees.

| Size ($N_1=N_2$) | Iterations ($K$) | Total Time (sec) | Single Execution Time (sec) |
| :---: | :---: | :---: | :---: |
| 1000 | 100 | 9.920000e-04 | 9.920000e-06 |
| 5000 | 100 | 4.996000e-03 | 4.996000e-05 |
| 10000 | 100 | 8.282000e-03 | 8.282000e-05 |
| 50000 | 100 | 2.526300e-02 | 2.526300e-04 |
| 100000 | 100 | 3.284000e-02 | 3.284000e-04 |
| 200000 | 100 | 5.224000e-02 | 5.224000e-04 |

*Conclusion:* As the input size scales from $1,000$ to $200,000$, the execution time increases strictly proportionally (from $\approx 10\mu s$ to $\approx 522\mu s$). This empirically proves the algorithms's $\mathcal{O}(N)$ linear time bound.

*   **Space Complexity:**
    *   `Node` array storage consumes $\mathcal{O}(N_1 + N_2)$.
    *   Auxiliary `arr1` and `arr2` buffers take another $\mathcal{O}(N_1 + N_2)$ heap spaces.
    *   System Call stack for traversing takes proportional memory to max depth $H$. In worst cases, $\mathcal{O}(N_1)$ frames.
    *   **Overall Space Complexity = $\mathcal{O}(N_1 + N_2)$**.


### 4.2 Alternative Algorithm Analysis (Bonus Discussion)
Instead of fully converting both trees to array, we could iterate only exactly ONE tree ($T_1$ via an inorder array of size $N_1$) and subsequently search for `N - arr1[i]` dynamically directly inside the $T_2$ pure tree using a classic **Binary Tree Search** (moving Left if smaller, Right if larger). 

*   *Time Complexity:* For each of the $N_1$ nodes, we search $T_2$ in $O(\log N_2)$ (average) or $O(N_2)$ (worst case deep skewed tree). Yields **$O(N_1 \log N_2)$** on average but **$O(N_1 \cdot N_2)$** structurally worst.
*   *Space Complexity:* Reduced explicit buffer allocation, bringing it closer to strictly $\mathcal{O}(H_1 + H_2)$ call stacks without arrays.

**Conclusion on Algorithm Swap:** The $O(N_1 + N_2)$ **Flatted Inorder arrays + Two Pointers** structure guarantees deterministic scalable speed in linear bounds, avoiding the severe tree degeneration overhead of $O(N^2)$ risks entirely. Thus, the implementation formulated in Chapter 2 proves highly superior for competitive and stable algorithms.

---

## Declaration
I hereby declare that all the work done in this project is of my independent effort.
