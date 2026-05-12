# Chapter3 Lists, Stacks, and Queues

## Abstract Data Type (ADT)

$\text{Data Type} = \text{\{Objects\}} \cup \text{\{Operations\}}$

**抽象数据类型 (ADT)**：把对象和操作的定义分离开，用户只知道对象数据是什么以及可以进行什么操作；而其内部数据的存储方法以及代码实现方法对于使用则隐藏。

## The List ADT

- **Objects**: $(\text{item}_0, \text{item}_1, \text{item}_2, \ldots, \text{item}_{n-1})$

- **操作**:
  - 求长度
  - 打印
  - 置空
  - 找第K个元素
  - 在第K个之后插入
  - 删除
  - 找后继节点
  - 找前驱节点

**数组实现的简单列表**

$$\text{array}[i]=\text{item}_i$$

- 最大尺寸需要估计
- 查找第K个元素需要 $O(1)$ 时间
- 插入和删除不仅需要 $O(n)$ 时间，还涉及大量的数据移动，耗时较多。

**链表实现的列表**

<!-- Standard Academic Table ("Three-Line Table") Style in HTML -->
<table style="width: 100%; border-collapse: collapse; border-top: 2px solid #e0e0e0; border-bottom: 2px solid #e0e0e0; text-align: left; background-color: transparent;">
    <thead>
        <tr style="border-bottom: 1px solid #e0e0e0;">
            <th style="padding: 10px; font-weight: bold;">Address</th>
            <th style="padding: 10px; font-weight: bold;">Data</th>
            <th style="padding: 10px; font-weight: bold;">Pointer</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td style="padding: 8px;">0010</td>
            <td style="padding: 8px;"><code>SUN</code></td>
            <td style="padding: 8px;"><code>1011</code></td>
        </tr>
        <tr>
            <td style="padding: 8px;">1011</td>
            <td style="padding: 8px;"><code>ZHAO</code></td>
            <td style="padding: 8px;"><code>0110</code></td>
        </tr>
        <tr>
            <td style="padding: 8px;">0110</td>
            <td style="padding: 8px;"><code>QIAN</code></td>
            <td style="padding: 8px;"><code>NULL</code></td>
        </tr>
    </tbody>
</table>

*初始化*
``` C
typedef struct list_node *list_ptr;
typedef struct list_node{
    char data[4];
    list_ptr next;
}
list_ptr ptr;
```

*链接节点*

``` C
list_ptr N1, N2;
N1 = (list_ptr)malloc(sizeof(struct list_node));
N2 = (list_ptr)malloc(sizeof(struct list_node));
N1->data = "ZHAO";
N2->data = "QIAN";
N1->next = N2;
N2->next = NULL;
ptr = N1;
```

*改进链表为双向循环链表*
``` C
typedef struct node *node_ptr;
typedef struct node{
    node_ptr llink;
    element item;
    node_ptr rlink;
}
```

## 列表的应用

**The Polynomial ADT**

- **Objects**: $P(x) = a_1 x^{e1} + a_{2} x^{e2} + \ldots + a_n x^{e_n}$
- **操作**:
  - 求多项式的次数
  - 两个多项式的加法
  - 两个多项式的减法
  - 两个多项式的乘法
  - 多项式的求导


```C
typedef struct poly_node *poly_ptr;
typedef struct poly_node{
    int Coefficient;
    int Exponent;
    poly_ptr Next;
}
typedef poly_ptr a;
```

**多链表 (Multilists)**

多链表的诞生主要是为了解决多对多的问题，比如需要存储学生-课程选课系统的信息存储。多链表的核心是：**一个节点可以属于多个不同的链表**，通过为节点设置多个指针域，分别指向不同链表的后记节点，实现多对多关系的双向关联存储。

针对学生节点和课程节点：
- 学生节点：数据域（ID）+ 指针域 （只想该学生选的下一门课程节点）
- 课程节点：数据域（ID）+ 指针域 （指向选了该课程的下一位学生节点）

```C
// 完全对齐你的语法风格
typedef struct multi_node *multi_ptr;
typedef struct multi_node{
    // 数据域（存储节点真实信息）
    int student_id;   // 学生学号
    int course_id;    // 课程编号
    
    // 多链表核心：多个指针域（属于两条不同的链表）
    multi_ptr next_student;  // 指针1：指向【同一门课的下一个学生】
    multi_ptr next_course;   // 指针2：指向【同一个学生的下一门课】
};
// 重定义指针类型（同你的 typedef poly_ptr a;）
typedef multi_ptr Multilist;
```

## 游标实现的链表 (无指针)

普通的链表实现依赖编程语言的指针和动态内存管理函数，但部分编程语言不支持指针或动态内存管理存在效率损耗。

游标实现的核心目标是： 用数组模拟链表的指针操作，通过游标（数组下标）代替物理指针


```C
typedef struct cursor_node *cursor_ptr;
typedef struct cursor_node{
    int Element;     // 数据域（存真实数据）
    int Next;        // 游标！存数组下标，代替指针
}
typedef cursor_ptr Cursor;
```

## The Stack ADT

**ADT**

栈是后进先出 **LIFO** 的线性表，所有插入删除只在栈顶进行。


- **对象**: 一个有限的有序列表，包含零个或多个元素。
- **操作**:
  - Int IsEmpty(S)  --- 判断栈是否为空
  - Stack CreateStack() --- 创建一个空栈
  - DisposeStack(Stack S) --- 销毁一个栈
  - MakeEmpty(Stack S) --- 将一个栈置空
  - Push(ElementType X, Stack S) --- 将元素X压入栈S
  - ElementType Top(Stack S) --- 返回栈S的栈顶元素
  - Pop(Stack S) --- 将栈S的栈顶元素弹出

**链表实现的栈 (带头节点)**
**
*Push:*
- `TmpCell->Next = S->Next`
- `S->Next = TmpCell`

> 这里的S是一个header node，S->Next才是真正的栈顶元素，所以Push操作是将新节点插入到header node和原来栈顶元素之间。

*Top:*
- `return S->Next->Element`

*Pop:*
- `FirstCell = S->Next`
- `S->Next = S->Next->Next`
- `free(FirstCell)`

> Pop操作是将栈顶元素弹出，即将header node的Next指向原来栈顶元素的Next，然后释放原来栈顶元素的内存。

**数组实现的栈**

```C
struct StackRecord{
    int Capacity;   // 栈的容量
    int TopOfStack; // 栈顶元素的下标
    ElementType *Array; // 存储栈元素的数组
}
```

> 栈模型必须封装良好。也就是说，除了栈例程之外，您的代码的任何部分都不能尝试访问“数组”或“栈顶”变量。
> 在执行“入栈”或“出栈（栈顶）”操作之前，必须进行错误检查。

## Stack的应用

**Balancing Symbols**

检查表达式中的 `( )`、`{ }`、`[ ]` 是否匹配。

*算法逻辑*
- 初始化空栈 S
- 逐字符遍历表达式：
    - 如果遇到左符号直接入栈
    - 如果遇到右符号：若栈为空或者栈顶元素不匹配则匹配失败，若匹配成功则弹出栈顶元素
- 便利结束后：若栈为空则符号完全匹配；若栈非空则匹配失败

> 时间复杂度：$O(n)$，其中 $n$ 是表达式的长度。每个字符最多被访问两次（一次入栈，一次出栈）。

> 同时算法不需要读取完整表达式即可判断不匹配的情况，适合流式处理

**Postfix Evaluation**

*概念*
- 中缀表达式：`a + b * c - d / e`（运算符在操作数中间）
- 前缀表达式（波兰式）：`- + a * b c / d e`
- 后缀表达式（逆波兰式）：`a b c + * d e / -`

*算法逻辑*
- 初始化空栈 S
- 逐个token遍历后缀表达式：
    - 遇到操作数直接入栈
    - 遇到运算符：从栈顶弹出两个操作数（先弹出的为右操作数），进行运算后将结果入栈
- 遍历结束后，栈顶元素即为表达式的最终结果

> 时间复杂度：$O(n)$，其中 $n$ 是后缀表达式的长度。每个token被访问一次，且每个运算符对应的操作数弹出和结果入栈操作都是常数时间。
> 无需考虑运算符优先级

**中缀表达式转后缀表达式**

*规则*
- 操作数顺序不变
- 运算符优先级：高级运算符在后缀表达式中出现的更早
- 括号处理：括号内的表达式有限转换，括号本身不输出

我们需要定义优先级：

<!-- Standard Academic Table ("Three-Line Table") Style in HTML -->
<table style="width: 100%; border-collapse: collapse; border-top: 2px solid #e0e0e0; border-bottom: 2px solid #e0e0e0; text-align: left; background-color: transparent;">
    <thead>
        <tr style="border-bottom: 1px solid #e0e0e0;">
            <th style="padding: 10px; font-weight: bold;">符号</th>
            <th style="padding: 10px; font-weight: bold;">入栈优先级</th>
            <th style="padding: 10px; font-weight: bold;">出栈优先级</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td style="padding: 8px;"><code>+</code><code>-</code></td>
            <td style="padding: 8px;">1</td>
            <td style="padding: 8px;">1</td>
        </tr>
        <tr>
            <td style="padding: 8px;"><code>*</code><code>/</code></td>
            <td style="padding: 8px;">2</td>
            <td style="padding: 8px;">2</td>
        </tr>
        <tr>
            <td style="padding: 8px;"><code>^</code></td>
            <td style="padding: 8px;">4</td>
            <td style="padding: 8px;">3</td>
        </tr>
        <tr>
            <td style="padding: 8px;"><code>(</code></td>
            <td style="padding: 8px;">5</td>
            <td style="padding: 8px;">0</td>
        </tr>
        <tr>
            <td style="padding: 8px;"><code>)</code></td>
            <td style="padding: 8px;">0</td>
            <td style="padding: 8px;">-</td>
        </tr>
    </tbody>
</table>

*算法逻辑*
- 初始化空栈 S 和输出列表 O
- 逐个token遍历中缀表达式：
    - 遇到操作数直接添加到输出列表 O
    - 遇到左括号 `(` 直接入栈
    - 遇到右括号 `)`：弹出栈顶运算符并添加到输出列表 O，直到遇到左括号 `(`，将左括号弹出但不添加到输出列表
    - 遇到运算符：比较其与栈顶运算符的优先级，若栈顶运算符优先级更高或相等，则弹出栈顶运算符并添加到输出列表 O，重复此过程直到栈顶运算符优先级更低或栈为空，然后将当前运算符入栈
- 遍历结束后，将栈中剩余的运算符依次弹出并添加到输出列表 O
- 输出列表 O 即为转换后的后缀表达式

## The Queue ADT

队列的核心是先入先出（FIFO）。因此，队列是一个有序列表，其中插入发生在末尾，而删除发生在前端。

- **Objects**: A finite ordered list with zero or more elements.
- **Operations**:
    - Int IsEmpty(Q) --- 判断队列是否为空
    - Queue CreateQueue() --- 创建一个空队列
    - DisposeQueue(Queue Q) --- 销毁一个队列
    - MakeEmpty(Queue Q) --- 将一个队列置空
    - Enqueue(ElementType X, Queue Q) --- 将元素X入队列Q
    - ElementType Front(Queue Q) --- 返回队列Q的队首元素
    - Dequeue(Queue Q) --- 将队列Q的队首元素出队

**Array Implementation of Queue**

```C
struct QueueRecord{
    int Capacity;   // 队列的容量
    int Front;      // 队首元素的下标
    int Rear;       // 队尾元素的下标
    int Size;       // 队列当前元素个数
    ElementType *Array; // 存储队列元素的数组
```

> 普通数组实现队列会导致出现假溢出的情况，即队列没有真正存满，但是队尾指针已经到达数组末尾，无法继续入队。这是由于`Front`和`Rear`指针都只单项移动，出队释放的队头前的空闲空间无法被利用。
> 上述问题我们利用循环队列来解决问题

**Circular Implementation of Queue**

整体修改的逻辑是：
- 入队`Rear = (Rear + 1) % Capacity` 来实现循环
- 出队`Front = (Front + 1) % Capacity` 来实现循环

> 但是还有个问题，就是满队列和空队列的条件都是`Front == Rear`，因此我们需要牺牲一个空间来区分满队列和空队列的情况，即当`(Rear + 1) % Capacity == Front`时认为队列满。
> 队列的所有基础操作的时间复杂度都是$O(1)$，因为每个操作只涉及指针的移动和少量的计算，不需要遍历队列中的元素。