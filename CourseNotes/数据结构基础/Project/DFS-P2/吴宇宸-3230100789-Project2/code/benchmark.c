/**
 * @file benchmark.c
 * @brief Performance Profiling for Two-Pointer Array Search (O(N) Logic)
 * 
 * 本文件专用于测试双指针匹配算法在不同数据量级下的实际运行时间。
 * 为了避免在分配超大极限测试集（如 N=200000）的时候因为过深的系统递归树
 * 而直接导致 Stack Overflow（爆栈），我们这里直接跳过建树，
 * 伪造已经完成中序遍历后的合法有序数组直接喂给算法核心去跑。
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// 核心的双指针求解算法。与 main.c 中的逻辑完全一致。
void run_solver(int n1, int n2, long long *arr1, long long *arr2, long long N) {
    int i = 0, j = n2 - 1;
    int found = 0;
    long long last_A = 0;
    int has_last = 0;

    // 前后指针往中间逼近
    while (i < n1 && j >= 0) {
        long long sum = arr1[i] + arr2[j];
        if (sum == N) {
            found = 1;
            if (!has_last || arr1[i] != last_A) {
                last_A = arr1[i];
                has_last = 1;
            }
            i++;
            j--;
        } else if (sum < N) {
            i++;
        } else {
            j--;
        }
    }
}

int main() {
    // 设定的测试规模级数组（从 1000 规模跨度到题目能给出的最大输入限界 200,000）
    int sizes[] = {1000, 5000, 10000, 50000, 100000, 200000}; 
    int num_tests = sizeof(sizes) / sizeof(sizes[0]);
    
    // 直接打印 Markdown 格式的表格表头
    printf("| Size ($N_1=N_2$) | Iterations ($K$) | Total Time (sec) | Single Execution Time (sec) |\n");
    printf("| :---: | :---: | :---: | :---: |\n");
    
    // 遍历每一个规模
    for (int t = 0; t < num_tests; t++) {
        int n = sizes[t];
        long long *arr1 = (long long *)malloc(n * sizeof(long long));
        long long *arr2 = (long long *)malloc(n * sizeof(long long));
        
        // 伪造两棵树对应的中序遍历升序数组的填充
        for (int i = 0; i < n; i++) {
            arr1[i] = i * 2;
            arr2[i] = i * 2 + 1;
        }
        
        long long target_N = n * 2; // 随意指定一个肯定大量存在配对结果的目标值
        
        int K = 100; // 设定循环执行 100 次（放大运行耗时解决时钟不够精密的问题）
        clock_t start = clock();
        
        // 执行 K 次测速
        for (int k = 0; k < K; k++) {
            run_solver(n, n, arr1, arr2, target_N);
        }
        clock_t end = clock();
        
        // 计算真实的时间（包含总时间和单次时间）
        double total_time = (double)(end - start) / CLOCKS_PER_SEC;
        double single_time = total_time / K;
        
        // 输出当前规模的测试结果所在的列
        printf("| %d | %d | %e | %e |\n", n, K, total_time, single_time);
        
        free(arr1);
        free(arr2);
    }
    
    return 0;
}
