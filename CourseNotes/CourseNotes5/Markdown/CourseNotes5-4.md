# Chapter5 Priority Queues (Heaps)

## ADT Model

**Objects**: A finite ordered list with zero or more elements.
**Operations**:
- `PriorityQueue Initialize(int Max Elements)`: 创建一个空的优先队列，最大元素个数为 Max Elements。
- `ElementType DeleteMin(PriorityQueue H)`: 从优先队列 H 中删除并返回最小元素。
- `void insert(ElementType X, PriorityQueue H)`: 将元素 X 插入优先队列 H 中。
- `int FindMin(PriorityQueue H)`: 返回优先队列 H 中最小元素的值。

---
## Simple Implementation

**Array**
- `Insertion`：将元素插入数组末尾，时间复杂度为 O(1)。
- `DeleteMin`：扫描整个数组找到最小元素并删除，时间复杂度为 O(n)。

**Linked List**
- `Insertion`：将元素插入链表末尾，时间复杂度为 O(1)。
- `DeleteMin`：扫描整个链表找到最小元素并删除，时间复杂度为 O(n)。

**Ordered Array**
- `Insertion`：将元素插入数组中合适的位置，时间复杂度为 O(n)。
- `DeleteMin`：直接删除数组第一个元素，时间复杂度为 O(1)。

**Ordered Linked List**
- `Insertion`：将元素插入链表中合适的位置，时间复杂度为 O(n)。
- `DeleteMin`：直接删除链表第一个元素，时间复杂度为 O(1)。

## Binary Heap

**Stucture Property**: 一棵具有 n 个节点和高度 h 的二叉树是完全二叉树，当且仅当其节点与高度为 h 的完美二叉树中编号从 1 到 n 的节点相对应。
- 一个高度为h的完全二叉树有 $2^h$ 到 $2^{h+1}-1$ 个节点。($h=\lfloor \log_2 n \rfloor$)

**Lemma**: 在一个拥有n个节点的完全二叉树中，节点编号为 $1, 2, \ldots, n$，则对于每个节点 $i$：
- 父节点编号为 $\lfloor i/2 \rfloor$。
- 左子节点编号为 $2i$。
- 右子节点编号为 $2i + 1$。

```C
PriorityQueue  Initialize( int  MaxElements ) 
{ 
     PriorityQueue  H; 
     if ( MaxElements < MinPQSize ) 
	return  Error( "Priority queue size is too small" ); 
     H = malloc( sizeof ( struct HeapStruct ) ); 
     if ( H ==NULL ) 
	return  FatalError( "Out of space!!!" ); 
     /* Allocate the array plus one extra for sentinel */ 
     H->Elements = malloc(( MaxElements + 1 ) * sizeof( ElementType )); 
     if ( H->Elements == NULL ) 
	return  FatalError( "Out of space!!!" ); 
     H->Capacity = MaxElements; 
     H->Size = 0; 
     H->Elements[ 0 ] = MinData;  /* set the sentinel */
     return  H; 
}
```

**Heap Order Priority**: 最小树的每个节点的键值都不大于其子节点（如果有）的键值。最小堆是一种完全二叉树，同时也是一棵最小树。

![alt text](PIC/PIC4-1.png)

**Basic Heap Operations**:

`Insert`：将元素插入堆中，时间复杂度为 O(log n)。

```C
/* H->Element[ 0 ] is a sentinel */ 
void  Insert( ElementType  X,  PriorityQueue  H ) 
{ 
     int  i; 

     if ( IsFull( H ) ) { 
	Error( "Priority queue is full" ); 
	return; 
     } 

     for ( i = ++H->Size; H->Elements[ i / 2 ] > X; i /= 2 ) 
	H->Elements[ i ] = H->Elements[ i / 2 ]; 

     H->Elements[ i ] = X; 
}
```

- `DeleteMin`：删除堆中的最小元素，时间复杂度为 O(log n)。

```C
ElementType  DeleteMin( PriorityQueue  H ) 
{ 
    int  i, Child; 
    ElementType  MinElement, LastElement; 
    if ( IsEmpty( H ) ) { 
         Error( "Priority queue is empty" ); 
         return  H->Elements[ 0 ];   } 
    MinElement = H->Elements[ 1 ];  /* save the min element */
    LastElement = H->Elements[ H->Size-- ];  /* take last and reset size */
    for ( i = 1; i * 2 <= H->Size; i = Child ) {  /* Find smaller child */ 
         Child = i * 2; 
         if (Child != H->Size && H->Elements[Child+1] < H->Elements[Child]) 
	       Child++;     
         if ( LastElement > H->Elements[ Child ] )   /* Percolate one level */ 
	       H->Elements[ i ] = H->Elements[ Child ]; 
         else     break;   /* find the proper position */
    } 
    H->Elements[ i ] = LastElement; 
    return  MinElement; 
}
```