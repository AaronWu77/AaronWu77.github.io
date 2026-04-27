# Homework of Chapter 3

## Problem 3-7
**Solution:**
The states cycle through 16 intervals (0000 to 1111). We partition the 80 seconds into 16 intervals of 5 seconds each.
- GNS (Green North/South) is on for 30s (6 intervals).
- YNS (Yellow North/South) is on for 5s (1 interval).
- RNS (Red North/South) is on for 45s (9 intervals).
- REW (Red East/West) is on for 45s (9 intervals). (Overlaps with RNS for 5s, 1 interval).
- GEW (Green East/West) is on for 30s (6 intervals).
- YEW (Yellow East/West) is on for 5s (1 interval).

Using K-maps or truth tables for the inputs $A, B, C, D$ to map to the 6 light outputs, we can minimize the logic to AND/OR gates and inverters.

## Problem 3-8
**Solution:**
A 3-bit input $X = X_2 X_1 X_0$ represents values from 0 to 7. The output $Y = X^2$ can range from 0 to 49 (requiring 6 bits $Y_5 Y_4 Y_3 Y_2 Y_1 Y_0$).
Truth table maps:
- 000 -> 000000
- 001 -> 000001
- 010 -> 000100
- 011 -> 001001
- 100 -> 010000
- 101 -> 011001
- 110 -> 100100
- 111 -> 110001

Boolean equations can be simplified from this truth table for each output bit $Y_i$.

## Problem 3-11
**Solution:**
Inputs: $PS, LS, RS, RR$
Outputs: $PL, LL, RL$
**Operation rules mapped to logic:**
1. $PL = PS$
2. $LL = \overline{PS} \cdot \overline{RS} \cdot LS + \overline{PS} \cdot LS \cdot RS \cdot RR$
3. $RL = \overline{PS} \cdot \overline{LS} \cdot RS + \overline{PS} \cdot LS \cdot RS \cdot \overline{RR}$

## Problem 3-13
**Solution:**
Given equations:
$$F = A(C\overline{E} + DE) + \overline{A}D$$
$$G = B(C\overline{E} + DE) + \overline{B}C$$
Let $H(X, Y, Z) = X(C\overline{E} + DE)$. We can map the instances of the hierarchical component to formulate the overall schematic block diagram using three copies of the component.

## Problem 3-14
**Solution:**
Hierarchical component: $H = \overline{X}Y + XZ$
Using Shannon's Expansion on $G = \overline{A}\overline{B}C + \overline{A}BD + A\overline{B}\overline{C} + AB\overline{D}$:
Evaluate $G$ at $A=0$ and $A=1$, configuring inputs to the $H$ block.

## Problem 3-16
**Solution:**
Technology mapping to NAND gates for Figure 3-54. Replace each AND, OR, and NOT gate with their NAND equivalents (e.g., an OR gate is a NAND gate with inverted inputs, which can be optimized with neighboring NANDs to cancel out double inversions).

## Problem 3-27
**Solution:**
Inputs: $S_i$ (sensors), $M$ (master), $A$ (alarm), $L$ (lights), $P$ (police).
Outputs: $A_{out}, L_{out}, V_{out}, C_{out}$.
Intrusion detected condition: $I = \overline{S_0} + \overline{S_1} + \overline{S_2} + \overline{S_3} + \overline{S_4} + \overline{S_5}$.
If $(I \cdot \overline{M}) == 1$:
$A_{out} = A$
$L_{out} = L$
$V_{out} = 1$
$C_{out} = P$
Else outputs are disabled (0).

## Problem 3-28
**Solution:**
A 4-to-16 line decoder uses a most significant bit (MSB) to select between two 3-to-8 line decoders. The outputs of the 3-to-8 decoders are gated with the MSB (or its complement) using the 16 2-input AND gates.

## Problem 3-29
**Solution:**
A 4-to-16 line decoder using five 2-to-4 line decoders with enable.
- The first 2-to-4 decoder takes the two MSBs and its 4 outputs connect to the enable pins of the remaining four 2-to-4 decoders.
- The other four 2-to-4 decoders share the two LSBs as their inputs.

## Problem 3-37
**Solution:**
(a) 8-to-1 multiplexer using 3-to-8 decoder and 8x2 AND-OR: The 3 selection lines feed into the 3-to-8 decoder. The 8 outputs of the decoder are each ANDed with the corresponding 8 data input lines, and all are summed together in an 8-input OR gate.

## Problem 3-44
**Solution:**
$F_1 = \overline{X} + \overline{Z} + XYZ$
$F_2 = \overline{X} + \overline{Z} + \overline{X}YZ$
$F_3 = \overline{X}\overline{Y}Z + \overline{X} + \overline{Z}$
Use a 3-to-8 line decoder for variables X, Y, Z. Convert equations to sum of minterms and attach corresponding decoder outputs to an external OR gate for each function.

## Problem 3-47
**Solution:**
$F(A,B,C,D) = \Sigma m(1,3,4,11,12,13,14,15)$
Using a 4-to-1 MUX with $A, B$ as selection lines ($S_1=A, S_0=B$).
- $AB = 00$: $F(0,0,C,D) = \Sigma m(1,3) = \overline{C}D + CD = D$
- $AB = 01$: $F(0,1,C,D) = \Sigma m(4) = \overline{C}\overline{D}$
- $AB = 10$: $F(1,0,C,D) = \Sigma m(11) = CD$
- $AB = 11$: $F(1,1,C,D) = \Sigma m(12,13,14,15) = 1$
These four expressions feed into the $I_0, I_1, I_2, I_3$ lines of the MUX.

## Problem 3-50
**Solution:**
Verify the circuit in Figure 3-58 implements a full adder:
Trace the Boolean logic for outputs $S_i$ and $C_{i+1}$. By analyzing the XOR and NAND/AND configurations, the results will reduce to $S_i = A_i \oplus B_i \oplus C_i$ and $C_{i+1} = A_i B_i + C_i(A_i \oplus B_i)$, which are standard full adder equations.

## Problem 3-51
**Solution:**
Obtain 1s and 2s complements:
- `10011100`: 1s = `01100011`, 2s = `01100100`
- `10011101`: 1s = `01100010`, 2s = `01100011`
- `10101000`: 1s = `01010111`, 2s = `01011000`
- `00000000`: 1s = `11111111`, 2s = `00000000` (carry thrown away)
- `10000000`: 1s = `01111111`, 2s = `10000000`

## Problem 3-52
**Solution:**
Subtraction with 2s complement of subtrahend ($M - N = M + \text{2s\_comp}(N)$):
(a) `11010 - 10001` $\rightarrow 11010 + 01111 = 101001 \rightarrow$ Result: `01001`
(b) `11110 - 01110` $\rightarrow 11110 + 10010 = 110000 \rightarrow$ Result: `10000`
(c) `1111110 - 1111110` $\rightarrow 1111110 + 0000010 = 10000000 \rightarrow$ Result: `0000000`
(d) `101001 - 000101` $\rightarrow 101001 + 111011 = 1100100 \rightarrow$ Result: `100100`

## Problem 3-59
**Solution:**
Comparisons $A, B$ (unsigned). Output $X = 1$ if $A < B$.
This is essentially a borrow-out calculation for a subtraction $A - B$. We can implement it using subtractor logic where the final Borrow-Out signal equates to $X$. Alternatively, define standard magnitude comparator equations for $A < B$.
