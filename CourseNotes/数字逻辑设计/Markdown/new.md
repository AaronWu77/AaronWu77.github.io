# Homework of Chapter 3 (Temp)

## Problem 3-7

A traffic light control at a simple intersection uses a binary counter to produce the following sequence of combinations on lines $A, B, C,$ and $D$:

$$
0000, 0001, 0011, 0010, 0110, 0111, 0101, 0100, 1100, 1101, 1111, 1110, 1010, 1011, 1001, 1000
$$

After 1000, the sequence repeats, beginning again with 0000, forever. Each combination is present for 5 seconds before the next one appears. These lines drive combinational logic with outputs to lamps RNS (red-north/south), YNS (yellow-north/south), GNS (green-north/south), REW (red-east/west), YEW (yellow-east/west), and GEW (green-east/west). The lamp controlled by each output is ON for a 1 applied and OFF for a 0 applied. For a given direction, assume that green is on for 30 seconds, yellow for 5 seconds, and red for 45 seconds. (The red intervals overlap for 5 seconds.) Divide the 80 seconds available for the cycle through the 16 combinations into 16 intervals and determine which lamps should be lit in each interval based on expected driver behavior. Assume that, for interval 0000, a change has just occurred and that $GNS = 1$, $REW = 1$, and all other outputs are 0. Design the logic to produce the six outputs using AND and OR gates and inverters.

---

## Problem 3-8

Design a combinational circuit that accepts a 3-bit number and generates a 6-bit binary number output equal to the square of the input number.

---

## Problem 3-11

A traffic metering system for controlling the release of traffic from an entrance ramp onto a superhighway has the following specifications for a part of its controller. There are three parallel metering lanes, each with its own stop (red)-go (green) light. One of these lanes, the car pool lane, is given priority for a green light over the other two lanes. Otherwise, a "round robin" scheme in which the green lights alternate is used for the other two (left and right) lanes. The part of the controller that determines which light is to be green (rather than red) is to be designed. The specifications for the controller follow:

**Inputs**

- $PS$: Car pool lane sensor (car present — 1; car absent — 0)
- $LS$: Left lane sensor (car present — 1; car absent — 0)
- $RS$: Right lane sensor (car present — 1; car absent — 0)
- $RR$: Round robin signal (select left — 1; select right — 0)

**Outputs**

- $PL$: Car pool lane light (green — 1; red — 0)
- $LL$: Left lane light (green — 1; red — 0)
- $RL$: Right lane light (green — 1; red — 0)

**Operation**

1. If there is a car in the car pool lane, $PL$ is 1.
2. If there are no cars in the car pool lane and the right lane, and there is a car in the left lane, $LL$ is 1.
3. If there are no cars in the car pool lane and in the left lane, and there is a car in the right lane, $RL$ is 1.
4. If there is no car in the car pool lane, there are cars in both the left and right lanes, and $RR$ is 1, then $LL = 1$.
5. If there is no car in the car pool lane, there are cars in both the left and right lanes, and $RR$ is 0, then $RL = 1$.
6. If any $PL$, $LL$, or $RL$ is not specified to be 1 above, then it has value 0.

**(a)** Find the truth table for the controller part.

**(b)** Find a minimum multiple-level gate implementation with minimum gate-input cost using AND gates, OR gates, and inverters.

---

## Problem 3-13

Design a circuit to implement the following pair of Boolean equations:

$$
F = A(C\overline{E} + DE) + \overline{A}D
$$

$$
G = B(C\overline{E} + DE) + \overline{B}C
$$

To simplify drawing the schematic, the circuit is to use a hierarchy based on the factoring shown in the equation. Three instances (copies) of a single hierarchical circuit component made up of two AND gates, an OR gate, and an inverter are to be used. Draw the logic diagram for the hierarchical component and for the overall circuit diagram using a symbol for the hierarchical component.

---

## Problem 3-14

A hierarchical component with the function is to be used along with inverters to implement the following equation:

$$
H = \overline{X}Y + XZ
$$

$$
G = \overline{A}\,\overline{B}C + \overline{A}BD + A\overline{B}\,\overline{C} + AB\overline{D}
$$

The overall circuit can be obtained by using Shannon's expansion theorem,

$$
F = \overline{X} \cdot F_0(X) + X \cdot F_1(X)
$$

where $F_0(X)$ is $F$ evaluated with variable $X = 0$ and $F_1(X)$ is $F$ evaluated with variable $X = 1$. This expansion $F$ can be implemented with function $H$ by letting $Y = F_0$ and $Z = F_1$. The expansion theorem can then be applied to each of $F_0$ and $F_1$ using a variable in each, preferably one that appears in both true and complemented form. The process can then be repeated until all $F_i$'s are single literals or constants. For $G$, use $X = A$ to find $G_0$ and $G_1$ and then use $X = B$ for $G_0$ and $G_1$. Draw the top-level diagram for $G$ using $H$ as a hierarchical component.

---

## Problem 3-16

Perform technology mapping to NAND gates for the circuit in Figure 3-54. Use cell types selected from: Inverter ($n = 1$), 2NAND, 3NAND, and 4NAND, as defined at the beginning of Section 3-2.

---

## Problem 3-27

A home security system has a master switch that is used to enable an alarm, lights, video cameras, and a call to local police in the event one or more of six sets of sensors detects an intrusion. In addition there are separate switches to enable and disable the alarm, lights, and the call to local police. The inputs, outputs, and operation of the enabling logic are specified as follows:

**Inputs**

- $S_i,\ i = 0,1,2,3,4,5$: signals from six sensor sets (0 = intrusion detected, 1 = no intrusion detected)
- $M$: master switch (0 = security system enabled, 1 = security system disabled)
- $A$: alarm switch (0 = alarm disabled, 1 = alarm enabled)
- $L$: light switch (0 = lights disabled, 1 = lights enabled)
- $P$: police switch (0 = police call disabled, 1 = police call enabled)

**Outputs**

- $A$: alarm (0 = alarm on, 1 = alarm off)
- $L$: lights (0 = lights on, 1 = lights off)
- $V$: video cameras (0 = video cameras off, 1 = video cameras on)
- $C$: call to police (0 = call off, 1 = call on)

**Operation**

If one or more of the sets of sensors detect an intrusion and the security system is enabled, then outputs activate based on the outputs of the remaining switches. Otherwise, all outputs are disabled.

---

## Problem 3-28

Design a 4-to-16-line decoder using two 3-to-8-line decoders and 16 2-input AND gates.

---

## Problem 3-29

Design a 4-to-16-line decoder with enable using five 2-to-4-line decoders with enable as shown in Figure 3-16.

---

## Problem 3-37

**(a)** Design an 8-to-1-line multiplexer using a 3-to-8-line decoder and an $8 \times 2$ AND-OR.

**(b)** Repeat part (a), using two 4-to-1-line multiplexers and one 2-to-1-line multiplexer.

---

## Problem 3-44

A combinational circuit is defined by the following three Boolean functions:

$$
F_1 = \overline{X + Z} + XYZ
$$

$$
F_2 = \overline{X + Z} + \overline{X}YZ
$$

$$
F_3 = \overline{X}YZ + \overline{X + Z}
$$

Design the circuit with a decoder and external OR gates.

---

## Problem 3-47

Implement the Boolean function

$$
F(A,B,C,D) = \Sigma m(1,3,4,11,12,13,14,15)
$$

with a 4-to-1-line multiplexer and external gates. Connect inputs $A$ and $B$ to the selection lines. The input requirements for the four data lines will be a function of the variables $C$ and $D$. The values of these variables are obtained by expressing $F$ as a function of $C$ and $D$ for each of the four cases when $AB = 00,\ 01,\ 10,$ and $11$. These functions must be implemented with external gates.

---

## Problem 3-50

The logic diagram of the first stage of a 4-bit adder, as implemented in integrated circuit type 74283, is shown in Figure 3-58. Verify that the circuit implements a full adder.

---

## Problem 3-51

Obtain the 1s and 2s complements of the following unsigned binary numbers:

$$
10011100,\quad 10011101,\quad 10101000,\quad 00000000,\quad \text{and}\quad 10000000.
$$

---

## Problem 3-52

Perform the indicated subtraction with the following unsigned binary numbers by taking the 2s complement of the subtrahend:

**(a)** $11010 - 10001$

**(b)** $11110 - 1110$

**(c)** $1111110 - 1111110$

**(d)** $101001 - 101$

---

## Problem 3-59

Design a combinational circuit that compares two 4-bit unsigned numbers $A$ and $B$ to see whether $B$ is greater than $A$. The circuit has one output $X$, so that $X = 1$ if $A < B$ and $X = 0$ if $A \geq B$.
