# 7.2 IR树表达式翻译练习

**题目**：
Translate each of these expressions into IR trees, but using the Ex, Nx, and Cx constructors as appropriate. In each case, just draw pictures of the trees; an Ex tree will be a Tree exp, an Nx tree will be a Tree stm, and a Cx tree will be a stm with holes labeled true and false into which labels can later be placed.
```
a. a+5  
b. output := concat(output, s), as it appears on line 8 of Program 6.3.  
c. b[i+1] := 0  
d. (c := a+1; c*c)  
e. while a>0 do a := a-1  
f. a<b moves a 1 or 0 into some newly defined temporary, and whose right-hand side is the temporary.  
g. if a then b else c, where a is an integer variable (true if ≠ 0).  
h. a := x+y  
i. if a<b then a else b  
j. if a<b then c:=a else c:=b
```

---

## 解答

### a. a+5
- **类型**：Ex
- **IR树**：
```
	+
   / \
  a   5
```
- **说明**：二元加法表达式，直接用Ex包装BINOP节点。

### b. output := concat(output, s)
- **类型**：Nx
- **IR树**：
```
	MOVE
   /    \
output  CALL
		 /   \
   concat   args
		   /    \
	  output    s
```
- **说明**：赋值语句，左边为output，右边为concat函数调用。

### c. b[i+1] := 0
- **类型**：Nx
- **IR树**：
```
	MOVE
   /    \
MEM      0
 |
 +
/ \
b  +
	/ \
   i   1
```
- **说明**：数组赋值，MEM表示内存写，地址为b+(i+1)。

### d. (c := a+1; c*c)
- **类型**：Ex
- **IR树**：
```
	ESEQ
   /    \
MOVE    *
/   \   / \
c   +  c   c
	/ \
   a   1
```
- **说明**：ESEQ先执行MOVE，再计算c*c。

### e. while a>0 do a := a-1
- **类型**：Nx
- **IR树**：
```
	WHILE
   /     \
  >      MOVE
 / \    /   \
a   0   a   -
			  / \
			 a   1
```
- **说明**：条件为a>0，循环体为a := a-1。

### f. a<b moves a 1 or 0 into some newly defined temporary
- **类型**：Ex
- **IR树**：
```
	ESEQ
   /    \
MOVE    TEMP
/   \     |
t   1/0   t
```
- **说明**：用条件跳转和MOVE实现布尔值赋给临时变量。

### g. if a then b else c (a为int变量)
- **类型**：Ex
- **IR树**：
```
	IF
   / | \
a  b  c
```
- **说明**：条件表达式，a≠0为真。

### h. a := x+y
- **类型**：Nx
- **IR树**：
```
	MOVE
   /    \
a      +
	   / \
	  x   y
```
- **说明**：赋值语句。

### i. if a<b then a else b
- **类型**：Ex
- **IR树**：
```
	IF
   / | \
 <  a  b
/ \
a   b
```
- **说明**：条件表达式。

### j. if a<b then c:=a else c:=b
- **类型**：Nx
- **IR树**：
```
	IF
   / | \
 <  MOVE  MOVE
/ \   / \  / \
a  b c  a c  b
```
- **说明**：条件语句，分支为赋值。

---
