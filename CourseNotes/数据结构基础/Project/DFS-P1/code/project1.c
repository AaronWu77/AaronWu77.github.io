#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/* 注意：CLK_TCK 在某些现代编译器中可能已被弃用，
   CLOCKS_PER_SEC 是每秒滴答数的标准 POSIX 宏。
   如果没有定义 CLK_TCK，我们将其定义为 CLOCKS_PER_SEC。 */
#ifndef CLK_TCK
#define CLK_TCK CLOCKS_PER_SEC
#endif

/* =========================================================
   1. 二分查找 (迭代版本)
   ========================================================= */
int binarySearchIterative(int *arr, int n, int target) {
    // 搜索空间的左右边界
    int left = 0, right = n - 1;
    while (left <= right) {
        // 使用 left + (right - left) / 2 而不是 (left + right) / 2 
        // 以防止 left 和 right 很大时发生整数溢出。
        int mid = left + (right - left) / 2;
        
        // 找到目标元素
        if (arr[mid] == target) return mid;
        
        // 根据比较结果调整边界
        if (arr[mid] < target) left = mid + 1;
        else right = mid - 1;
    }
    return -1; // 未找到
}

/* =========================================================
   2. 二分查找 (递归版本)
   ========================================================= */
int binarySearchRecursive(int *arr, int left, int right, int target) {
    if (left > right) return -1; // 基本情况：未找到
    
    // 安全地计算 mid 以避免整数溢出
    int mid = left + (right - left) / 2;
    if (arr[mid] == target) return mid;
    
    // 在适当的一半数组中递归搜索
    if (arr[mid] < target) 
        return binarySearchRecursive(arr, mid + 1, right, target);
    else 
        return binarySearchRecursive(arr, left, mid - 1, target);
}

/* 包装函数，用于统一函数签名 */
int binarySearchRecWrapper(int *arr, int n, int target) {
    return binarySearchRecursive(arr, 0, n - 1, target);
}

/* =========================================================
   3. 顺序查找 (迭代版本)
   ========================================================= */
int sequentialSearchIterative(int *arr, int n, int target) {
    for (int i = 0; i < n; i++) {
        if (arr[i] == target) return i;
    }
    return -1; // 未找到
}

/* =========================================================
   4. 顺序查找 (递归版本)
   ========================================================= */
int sequentialSearchRecursive(int *arr, int index, int n, int target) {
    if (index >= n) return -1; // 基本情况：到达数组末尾
    if (arr[index] == target) return index;
    
    return sequentialSearchRecursive(arr, index + 1, n, target);
}

/* 包装函数，用于统一函数签名 */
int sequentialSearchRecWrapper(int *arr, int n, int target) {
    return sequentialSearchRecursive(arr, 0, n, target);
}


/* =========================================================
   测试和测量工具
   ========================================================= */

/* 函数指针类型，允许传入不同的搜索算法 */
typedef int (*SearchFunc)(int *, int, int);

/* 
 * measurePerformance
 * 评估给定搜索算法的最坏情况时间复杂度。
 * 保证至少运行 10 个滴答周期以确保计时准确。
 */
void measurePerformance(SearchFunc func, int *arr, int n, const char *name) {
    int target = n; // 因为数组包含 0 到 n-1 的元素，寻找 'n' 确保了最坏情况。
    long K = 10;    // 初始迭代次数
    clock_t start, stop;
    double ticks;

    // 动态增加 K，直到消耗的滴答数 >= 10
    while (1) {
        start = clock();
        for (long i = 0; i < K; i++) {
            func(arr, n, target);
        }
        stop = clock();
        
        ticks = (double)(stop - start);
        if (ticks >= 10.0) {
            break;
        }
        K *= 10; // 如果滴答数太小，将 K 扩大 10 倍
    }

    double totalTime = ticks / CLK_TCK; // 总耗时（秒）
    double duration = totalTime / K;    // 单次函数执行运行时间

    // 打印表格行
    printf("%-30s | %6d | %12ld | %8.0f | %15.6e | %15.6e\n", 
           name, n, K, ticks, totalTime, duration);
}

/* =========================================================
   主执行逻辑块
   ========================================================= */
int main() {
    // 根据项目要求定义的 N 值
    int N_values[] = {100, 500, 1000, 2000, 4000, 6000, 8000, 10000};
    int num_N = sizeof(N_values) / sizeof(N_values[0]);

    // 表头
    printf("%-30s | %6s | %12s | %8s | %15s | %15s\n", 
           "Algorithm", "N", "Iterations(K)", "Ticks", "Total Time(sec)", "Duration(sec)");
    printf("------------------------------------------------------------------------------------------------------\n");

    // 遍历所有必需的 N 值测试用例
    for (int i = 0; i < num_N; i++) {
        int n = N_values[i];
        
        // 动态分配并初始化内存，以防止大 N 值时发生栈溢出
        int *arr = (int *)malloc(n * sizeof(int));
        if (arr == NULL) {
            printf("内存分配失败！\n");
            return -1;
        }
        // 填充从 0 到 N-1 的有序整数
        for (int j = 0; j < n; j++) {
            arr[j] = j;
        }

        // 在最坏情况约束下运行性能测量
        measurePerformance(binarySearchIterative, arr, n, "Binary Search (Iterative)");
        measurePerformance(binarySearchRecWrapper, arr, n, "Binary Search (Recursive)");
        measurePerformance(sequentialSearchIterative, arr, n, "Sequential Search (Iterative)");
        measurePerformance(sequentialSearchRecWrapper, arr, n, "Sequential Search (Recursive)");
        
        printf("------------------------------------------------------------------------------------------------------\n");
        free(arr); // 清理分配的内存
    }

    return 0;
}
