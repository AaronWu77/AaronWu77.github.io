import matplotlib.pyplot as plt

# 来自性能测量表的数据
N = [100, 500, 1000, 2000, 4000, 6000, 8000, 10000]

# 各算法的执行时间(秒)
binary_iterative = [1.3e-08, 1.6e-08, 1.8e-08, 2.1e-08, 1.1e-07, 2.7e-08, 3.2e-08, 4.6e-08]
binary_recursive = [2.6e-08, 3.6e-08, 5.2e-08, 4.8e-08, 1.1e-07, 5.9e-08, 6.8e-08, 6.5e-08]
sequential_iterative = [6.4e-08, 3.6e-07, 7.0e-07, 4.5e-06, 9.2e-06, 3.6e-06, 5.8e-06, 6.0e-06]
sequential_recursive = [5.5e-07, 4.4e-06, 7.4e-06, 1.87e-05, 4.06e-05, 5.02e-05, 7.10e-05, 7.60e-05]

# 创建图表
plt.figure(figsize=(10, 6))

# 绘制带有标记的每条折线
plt.plot(N, binary_iterative, label='Binary Search (Iterative)', marker='o', linestyle='-', linewidth=2)
plt.plot(N, binary_recursive, label='Binary Search (Recursive)', marker='s', linestyle='-', linewidth=2)
plt.plot(N, sequential_iterative, label='Sequential Search (Iterative)', marker='^', linestyle='-', linewidth=2)
plt.plot(N, sequential_recursive, label='Sequential Search (Recursive)', marker='d', linestyle='-', linewidth=2)

# 图表配置
plt.title('Performance Comparison: N vs. Run Time (Worst Case)', fontsize=14, pad=15)
plt.xlabel('N (Array Size)', fontsize=12)
plt.ylabel('Duration in seconds (sec)', fontsize=12)
plt.grid(True, which="both", ls="--", alpha=0.6)
plt.legend(fontsize=11)

# 因为二分查找极快(10^-8)而顺序查找较慢(10^-5)，
# 使用对数刻度有助于在同一图表中清晰地可视化两者。
# 如果您希望使用线性刻度，可以注释掉下一行。
plt.yscale('log')

# 将图表作为图片保存在同级目录下
plt.savefig('plot.png', dpi=300, bbox_inches='tight')

print("Success: 'plot.png' has been generated.")
