/*
 * Normal-2 A+B with Binary Search Trees
 * 
 * 该程序用于解决给定两棵二叉搜索树（BST），在两棵树中各取一个节点值，使得它们的和等于目标值N的问题。
 * 
 * 核心算法思路：
 * 1. 树的表示：使用结构体数组，通过父节点索引建立左右孩子关系（考虑到BST性质：左小右大）。
 * 2. 中序遍历特性：对BST进行中序遍历能得到一个严格递增的有序数组。
 * 3. 双指针匹配：将两棵树分别中序遍历得到有序数组后，使用首尾双指针，在 O(N1 + N2) 的时间复杂度内找出所有和为N的配对。
 * 4. 前序遍历：通过标准的递归方式输出树的前序遍历序列。
 */

#include <stdio.h>
#include <stdlib.h>

// 二叉树节点结构体
typedef struct {
    long long key;  // 节点的键值
    int left;       // 左孩子在数组中的索引位置，-1表示无左孩子
    int right;      // 右孩子在数组中的索引位置，-1表示无右孩子
} Node;

// 查找树的根节点
// 遍历所有节点的父节点索引，父节点索引为 -1 的即为根节点
int find_root(int n, int *parent) {
    for (int i = 0; i < n; i++) {
        if (parent[i] == -1) return i;
    }
    return -1;
}

// 根据父节点索引数组和BST性质构建二叉树
// 若当前节点的值小于父节点的值，则为左孩子；否则为右孩子
void build_tree(int n, Node *nodes, int *parent) {
    for (int i = 0; i < n; i++) {
        nodes[i].left = -1;
        nodes[i].right = -1;
    }
    for (int i = 0; i < n; i++) {
        int p = parent[i];
        if (p != -1) {
            if (nodes[i].key < nodes[p].key) {
                nodes[p].left = i;
            } else {
                nodes[p].right = i;
            }
        }
    }
}

// 前序遍历二叉树并按格式输出
// 格式要求：节点间以1个空格分隔，行首行末无多余空格
void preorder(int root, Node *nodes, int *is_first) {
    if (root == -1) return;
    if (!(*is_first)) printf(" ");
    printf("%lld", nodes[root].key);
    *is_first = 0;
    preorder(nodes[root].left, nodes, is_first);
    preorder(nodes[root].right, nodes, is_first);
}

// 中序遍历二叉树并将节点值存入数组中
// 由于该树是BST，中序遍历的结果一定是递增排列的
void inorder(int root, Node *nodes, long long *arr, int *idx) {
    if (root == -1) return;
    inorder(nodes[root].left, nodes, arr, idx);
    arr[(*idx)++] = nodes[root].key; // 记录当前节点值
    inorder(nodes[root].right, nodes, arr, idx);
}

// 读取一棵树的完整输入并完成构建与中序遍历
void process_input_tree(int *n, Node **nodes, long long **arr, int *root) {
    if (scanf("%d", n) != 1) return;
    
    *nodes = NULL;
    *arr = NULL;
    *root = -1;
    
    if (*n > 0) {
        *nodes = (Node *)malloc(*n * sizeof(Node));
        int *parent = (int *)malloc(*n * sizeof(int));
        
        // 读取节点值与父节点索引
        for (int i = 0; i < *n; i++) {
            scanf("%lld %d", &((*nodes)[i].key), &parent[i]);
        }
        
        build_tree(*n, *nodes, parent);      // 构建树结构
        *root = find_root(*n, parent);       // 找到根节点
        
        *arr = (long long *)malloc(*n * sizeof(long long));
        int idx = 0;
        inorder(*root, *nodes, *arr, &idx);  // 获得递增的中序序列
        
        free(parent); // 及时释放父节点数组
    }
}

int main() {
    int n1, n2;
    Node *t1_nodes = NULL, *t2_nodes = NULL;
    long long *arr1 = NULL, *arr2 = NULL;
    int root1 = -1, root2 = -1;

    // 读取并处理两棵树 T1 和 T2
    process_input_tree(&n1, &t1_nodes, &arr1, &root1);
    process_input_tree(&n2, &t2_nodes, &arr2, &root2);

    // 读取目标值 N
    long long N;
    if (scanf("%lld", &N) != 1) return 0;

    // 双指针寻找和为 N 的配对组合 (A + B = N)
    // i 从 arr1 最小值往后遍历，j 从 arr2 最大值往前遍历
    int i = 0, j = n2 - 1;
    int found = 0;          // 是否找到至少一种解的标志
    long long last_A = 0;   // 用于去重，防止输出相同的配对
    int has_last = 0;       // 是否已经有上一次输出的标志

    while (i < n1 && j >= 0) {
        long long sum = arr1[i] + arr2[j];
        
        if (sum == N) {
            if (!found) {
                printf("true\n");
                found = 1;
            }
            // 要求按照 A 的升序排序输出，且相同的等式只输出一次
            if (!has_last || arr1[i] != last_A) {
                printf("%lld = %lld + %lld\n", N, arr1[i], arr2[j]);
                last_A = arr1[i];
                has_last = 1;
            }
            i++;
            j--;
        } else if (sum < N) {
            // 和小于 N，说明需要更大的值，前指针后移
            i++;
        } else {
            // 和大于 N，说明需要更小的值，后指针前移
            j--;
        }
    }

    // 若未找到任何解，输出 false
    if (!found) {
        printf("false\n");
    }

    // 输出 T1 和 T2 的前序遍历
    if (n1 > 0) {
        int is_first1 = 1;
        preorder(root1, t1_nodes, &is_first1);
        printf("\n");
    } else {
        printf("\n"); // 树为空则仅换行
    }

    if (n2 > 0) {
        int is_first2 = 1;
        preorder(root2, t2_nodes, &is_first2);
        printf("\n");
    } else {
        printf("\n"); // 树为空则仅换行
    }

    // 释放动态分配的内存
    if (t1_nodes) free(t1_nodes);
    if (arr1) free(arr1);
    if (t2_nodes) free(t2_nodes);
    if (arr2) free(arr2);

    return 0;
}
