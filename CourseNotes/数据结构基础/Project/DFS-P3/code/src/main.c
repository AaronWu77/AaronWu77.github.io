#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAXV 1005
#define MAXE 200005
#define INF 0x3f3f3f3f

/*
 * Problem:
 *   Check whether each permutation is a valid Dijkstra selecting order.
 * Input:
 *   Undirected connected weighted graph, then K permutations.
 * Output:
 *   Print "Yes" if a permutation can be produced by Dijkstra, otherwise "No".
 * Method:
 *   Force-select vertices in given order and verify min-dist rule step by step.
 * Complexity per query:
 *   O(N^2 + M) with adjacency list + linear scan minimum.
 */

/* 邻接表边结点：终点、权重、下一条边下标。 */
typedef struct {
    int to;
    int w;
    int next;
} Edge;

/* 图存储：无向图用两条有向边。 */
static Edge edges[MAXE];
static int head[MAXV];
static int edge_count = 0;

/* Dijkstra 模拟过程中的工作数组。 */
static int distv[MAXV];
static int visited[MAXV];
static int seq[MAXV];

/*
 * Notes:
 *   1) distv[x] is tentative shortest distance from current source to x.
 *   2) visited[x] indicates whether x has been fixed by Dijkstra process.
 *   3) seq[i] is the i-th vertex in the queried order.
 */

/* 向邻接表头插入一条有向边 u -> v。 */
static void add_edge(int u, int v, int w) {
    edges[edge_count].to = v;
    edges[edge_count].w = w;
    edges[edge_count].next = head[u];
    head[u] = edge_count++;
}

/*
 * 检查给定排列是否可能是某次 Dijkstra 的“选点顺序”。
 * 判定要点：每一步被选中的顶点，必须是当前未收录顶点中 dist 最小者之一。
 */
static int is_dijkstra_sequence(int n) {
    int i;
    int source = seq[0];

    /*
     * 核心判定语义：
     * 只要任何一步违反“当前最小 dist 选点”规则，就立即失败。
     */

    /* 每个查询都要重新初始化访问标记和距离数组。 */
    memset(visited, 0, sizeof(visited));
    /* 非源点初始为无穷大，符合 Dijkstra 初始化定义。 */
    for (i = 1; i <= n; ++i) {
        distv[i] = INF;
    }
    distv[source] = 0;

    /* 按给定序列强制“选点”。 */
    for (i = 0; i < n; ++i) {
        int u = seq[i];
        int v;
        int min_dist = INF;
        int e;

        /* i 表示第 i 次“取最小”操作。 */

        /* 出现重复顶点，必不是排列，直接非法。 */
        if (visited[u]) {
            return 0;
        }

        /* 找当前所有未访问顶点中的最小 dist。 */
        for (v = 1; v <= n; ++v) {
            if (!visited[v] && distv[v] < min_dist) {
                min_dist = distv[v];
            }
        }

        /* Dijkstra 本步只能选最小 dist 顶点（允许并列）。 */
        if (distv[u] != min_dist) {
            return 0;
        }

        /* u 被确认后，其最短路值不再变化。 */
        visited[u] = 1;

        /* 标准松弛操作：尝试缩短邻接点距离。 */
        /* 扫描 u 的邻接边，执行 relax。 */
        for (e = head[u]; e != -1; e = edges[e].next) {
            int to = edges[e].to;
            int w = edges[e].w;

            /* 仅对未确认顶点做松弛，避免重复更新已确定最短路顶点。 */
            if (!visited[to] && distv[u] + w < distv[to]) {
                distv[to] = distv[u] + w;
            }
        }
    }

    return 1;
}

int main(void) {
    int n, m;
    int i;
    int k;

    /* 主流程：读图 -> 读查询 -> 逐条输出 Yes/No。 */

    /* 读取顶点数和边数；输入不完整时直接结束。 */
    if (scanf("%d %d", &n, &m) != 2) {
        return 0;
    }

    /* 题目约束内，静态数组容量已经覆盖最大规模。 */

    /* 邻接表头初始化为 -1，表示空链。 */
    memset(head, -1, sizeof(head));

    /* 读入无向图：每条边拆成两条有向边。 */
    for (i = 0; i < m; ++i) {
        int u, v, w;
        scanf("%d %d %d", &u, &v, &w);

        /* 无向边拆两次插入。 */
        add_edge(u, v, w);
        add_edge(v, u, w);
    }

    /* 逐个查询：每行一个顶点排列。 */
    scanf("%d", &k);
    /* 每个查询独立判定，互不影响。 */
    while (k--) {
        for (i = 0; i < n; ++i) {
            scanf("%d", &seq[i]);
        }

        /* 输出是否为合法 Dijkstra 选点序列。 */
        if (is_dijkstra_sequence(n)) {
            puts("Yes");
        } else {
            puts("No");
        }
    }

    return 0;
}
