# Homework of Chapter 3

## Problem 3-7
A trafic light control at a simple intersection uses a binary counter to produce the following sequence of combinations on lines A, B, C, D: 0000, 0001, 0011, 0010, 0110, 0111, 0101, 0100, 1100, 1101, 1111, 1110, 1010, 1011, 1001, 1000. After 1000, the sequence repeats, beginning again with 0000,forever. Each combination is present for 5 seconds before the next one appears. These lines drive combinational logic with outputs to lamps RNS (red—north/south), YNS (yellow—north/south), GNS (green—north/south), REW (red—east/west), YEW (yellow—east/west), and GEW (green—east/west). The lamp controlled by each output is ON for a 1 applied and OFF for a 0 applied. For a given direction, assume that green is on for 30 seconds, yellow for 5 seconds, and red for 45 seconds. (The red intervals overlap for 5 seconds.) Divide the 80 seconds available for the cycle through the 16 combinations into 16 intervals and determine which lamps should be lit in each interval based on expected driver behavior. Assume that, for interval 0000, a change has just occurred and that GNS = 1, REW = 1, and all other outputs are 0. Design the logic to produce the six outputs using AND and OR gates and inverters.

## Problem 3-8
Design a combinational circuit that accepts a 3-bit number and generates a 6-bit binary number output equal to the square of the input number

## Problem 3-11
A Traffic metering system for controlling the release of traffic from an entrance ramp onto a superhighway has the following specifications for a part of its contrller. There are three parallel metering lanes, each with its own stop(red)-go(green) light. Oneof these lanes, the car pool lan, is given priority for a green light over the other two lanes. Otherwise, a "round robin" scheme in which the green lights alternate is used for the other twp (left and right) lanes. The part of the controller that determines which light is to be gren (rather than red) is to be designed. The specifications for the contrller follow:
```
Inputs
    PS  Car pool lane sensor (car present -1; car absent -0)
    LS  Left lane sensor (car present -1; car absent -0)
    RS  Right lane sensor (car present -1; car absent -0)
    RR  Roud robin signal (select left -1; select right -0)

Outputs
    PL  Car pool lane light (green —1; red —0)
    LL  Left lane light (green —1; red —0)
    RL  Right lane light (green —1; red —0)

Operation
    1. If there is a car in the car pool lane, PL is 1.
    2. If there are no cars in the car pool lane and the right lane, and there is a car in the left lane, LL is 1.
    3. If there are no cars in the car pool lane and in the left lane, and there is a car in the right lane, RL is 1.
    4. If there is no car in the car pool lane, there are cars in both the left and right lanes, and RR is 1, then LL=1.
    5. If there is no car in the car pool lane, there are cars in both the left and right lanes, and RR is 0, then RL=1.
    6. If any PL, LL, or RL is not speciied to be 1 above, then it has value 0
```
(a) Find the truth table for the controller part
(b) Find a minimum multiple-level gate implementation with minimum gate-input cost using AND gates, OR gates and inverters



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
