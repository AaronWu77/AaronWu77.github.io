# Homework of Chap 15 & 16

## Chapter 15

### T1 (15.7)
(a) Show that Eq. 15-13, the variation of pressure with altitude in the atmosphere (temperature assumed to be uniform), can be written in terms of density $\rho$ as

$$
\rho = \rho_0 e^{-h/a},
$$

where $\rho_0$ is the density at the ground ($h = 0$).

(b) Assume that the drag force $D$ due to the air on an object moving at speed $v$ is given by

$$
D = CA\rho v^2
$$

where $C$ is a constant, $A$ is the frontal cross-sectional area of the object, and $\rho$ is the local air density. Find the altitude at which the drag force on a rocket is a maximum if the rocket is launched vertically and moves with constant upward acceleration $a_r$.

**solution**
- (a) Isothermal atmosphere: $p \propto \rho$ (ideal gas), and Eq. 15-13 gives $p=p_0e^{-h/a}$, so
  $$\rho=\rho_0e^{-h/a}.$$
- (b) For constant upward acceleration from rest, $v^2=2a_r h$.
  $$D(h)=CA\rho_0e^{-h/a}(2a_r h)=Khe^{-h/a}.$$
  Maximize $f(h)=he^{-h/a}$: $f'(h)=e^{-h/a}(1-h/a)=0 \Rightarrow h=a$.

---

### T2 (15.8)
(a) Consider a container of fluid subject to a vertical upward acceleration $a$. Show that the pressure variation with depth in the fluid is given by

$$
p = \rho h(g + a),
$$

where $h$ is the depth and $\rho$ is the density.

(b) Show also that if the fluid as a whole undergoes a vertical downward acceleration $a$, the pressure at depth $h$ is given by

$$
p = \rho h(g - a).
$$

(c) What is the state of affairs in free fall?

**solution**
- In the non-inertial frame of the container, effective gravity is $g_{\text{eff}}$.
- (a) Upward acceleration: $g_{\text{eff}}=g+a$, so $dp/dh=\rho(g+a)$, hence $p=\rho h(g+a)$ (gauge).
- (b) Downward acceleration: $g_{\text{eff}}=g-a$, so $p=\rho h(g-a)$.
- (c) Free fall: $a=g \Rightarrow g_{\text{eff}}=0$, so no hydrostatic gradient ($dp/dh=0$), fluid is effectively weightless.

---

### T3 (15.11)
Show that the variation of pressure with altitude for a planetary atmosphere (assuming constant temperature) is

$$
p = p_0 e^{k(1/r - 1/R)}
$$

where $g$ is taken to vary as $1/r^2$ (with $r$ being the distance from the center of the planet), $p_0$ is the pressure at the surface, $R$ is the radius of the planet, and $k$ is a constant. Verify that this result reduces to Eq. 15-12 for locations close to the surface.

**solution**
- Hydrostatic + ideal gas (isothermal):
  $$\frac{dp}{p}=-\frac{\mu g(r)}{RT}\,dr,\quad g(r)=\frac{GM}{r^2}.$$
- Integrate from $(R,p_0)$ to $(r,p)$:
  $$\ln\frac{p}{p_0}=\frac{\mu GM}{RT}\left(\frac{1}{r}-\frac{1}{R}\right).$$
  Let $k=\mu GM/(RT)$:
  $$p=p_0e^{k(1/r-1/R)}.$$
- Near surface $r=R+h$ with $h\ll R$:
  $$\frac{1}{r}-\frac{1}{R}\approx-\frac{h}{R^2},$$
  so $p\approx p_0e^{-h/a}$ with $a=R^2/k$, i.e., Eq. 15-12 form.

---

### T4 (15.13)
A hollow spherical iron shell floats almost completely submerged in water (see Fig. 15-27). The outer diameter is $58.7\ \text{cm}$ and the density of iron is $7.87\ \text{g/cm}^3$. Find the inner diameter of the shell.

**solution**
- Nearly fully submerged: buoyancy equals weight,
  $$\rho_w V_o g=\rho_{Fe}(V_o-V_i)g.$$
- Thus
  $$\frac{V_i}{V_o}=1-\frac{\rho_w}{\rho_{Fe}}=1-\frac{1}{7.87}=0.8729.$$
- Radius ratio:
  $$\frac{r_i}{r_o}=\left(\frac{V_i}{V_o}\right)^{1/3}\approx0.9557.$$
- Inner diameter:
  $$D_i=0.9557\times 58.7\,\text{cm}\approx56.1\,\text{cm}. $$

---

### T5 (15.17)
You place a glass beaker, partially filled with water, in a sink (Fig. 15-29). It has a mass of $390\ \text{g}$ and an interior volume of $500\ \text{cm}^3$. You now start to fill the sink with water and you find, by experiment, that if the beaker is less than half full, it will float; but if it is more than half full, it remains on the bottom of the sink as the water rises to its rim. What is the density of the material of which the beaker is made?

**solution**
- Critical condition occurs at half full: neutral between floating and sinking.
- At threshold, total mass:
  $$m_{tot}=390+\frac{1}{2}(500)=640\,\text{g}.$$
- Corresponding displaced volume in water:
  $$V_{ext}=\frac{m_{tot}}{\rho_w}=640\,\text{cm}^3.$$
- Glass material volume:
  $$V_g=V_{ext}-V_{in}=640-500=140\,\text{cm}^3.$$
- Glass density:
  $$\rho_g=\frac{m_g}{V_g}=\frac{390}{140}\approx2.79\,\text{g/cm}^3.$$

---

## Chapter 16

### T6 (16.4)
A siphon is a device for removing liquid from a container that is not to be tipped. It operates as shown in Fig. 16-36. The tube must initially be filled, but once this has been done the liquid will flow until its level drops below the tube opening at $A$. The liquid has density $\rho$ and negligible viscosity.

(a) With what speed does the liquid emerge from the tube at $C$?

(b) What is the pressure in the liquid at the topmost point $B$?

(c) What is the greatest possible height $h$ that a siphon may lift water?

**solution**
- Take reservoir free surface as point $S$ ($p_S=p_{atm}$, $v_S\approx0$).
- (a) Bernoulli from $S$ to outlet $C$ (drop $h_2$):
  $$v_C=\sqrt{2gh_2}.$$
- (b) At crest $B$ (height $h_1$ above surface), with same tube speed $v\approx v_C$:
  $$p_B=p_{atm}-\rho gh_1-\frac{1}{2}\rho v^2
  =p_{atm}-\rho g(h_1+h_2).$$
- (c) Require $p_B\ge p_v$ (vapor pressure) to avoid cavitation:
  $$h_{1,\max}=\frac{p_{atm}-p_v}{\rho g}-h_2.$$
  Ideal upper bound for water is about $10\,\text{m}$ order.

---

### T7 (16.7)
A water intake at a storage reservoir (see Fig. 16-28) has a cross-sectional area of $7.60\ \text{ft}^2$. The water flows in at a speed of $1.33\ \text{ft/s}$. At the generator building $572\ \text{ft}$ below the intake point, the water flows out at $31.0\ \text{ft/s}$.

(a) Find the difference in pressure, in $\text{lb/in.}^2$, between inlet and outlet.

(b) Find the area of the outlet pipe.

The weight density of water is $62.4\ \text{lb/ft}^3$.

**solution**
- (b) Continuity:
  $$A_2=A_1\frac{v_1}{v_2}=7.60\frac{1.33}{31.0}\approx0.326\,\text{ft}^2.$$
- (a) Bernoulli (1=inlet, 2=outlet, $z_1-z_2=572\,\text{ft}$):
  $$p_2-p_1=\gamma\left[(z_1-z_2)+\frac{v_1^2-v_2^2}{2g}\right].$$
  With $\gamma=62.4\,\text{lb/ft}^3$, $g=32.2\,\text{ft/s}^2$:
  $$p_2-p_1\approx3.48\times10^4\,\text{lb/ft}^2\approx241\,\text{psi}.$$
  So outlet pressure is about $241\,\text{psi}$ higher than inlet.

---

### T8 (16.10)
Water is moving with a speed of $5.18\ \text{m/s}$ through a pipe with a cross-sectional area of $4.20\ \text{cm}^2$. The water gradually descends $9.66\ \text{m}$ as the pipe increases in area to $7.60\ \text{cm}^2$.

(a) What is the speed of flow at the lower level?

(b) The pressure at the upper level is $152\ \text{kPa}$; find the pressure at the lower level.

**solution**
- (a) Continuity:
  $$v_2=v_1\frac{A_1}{A_2}=5.18\frac{4.20}{7.60}\approx2.86\,\text{m/s}.$$
- (b) Bernoulli ($z_1-z_2=9.66\,\text{m}$):
  $$p_2=p_1+\frac{1}{2}\rho(v_1^2-v_2^2)+\rho g(z_1-z_2).$$
  Using $\rho=1000\,\text{kg/m}^3$:
  $$p_2\approx152\,\text{kPa}+9.3\,\text{kPa}+94.7\,\text{kPa}\approx256\,\text{kPa}. $$

---

### T9 (16.11)
In flows that are sharply curved, centrifugal effects are appreciable. Consider an element of fluid that is moving with speed $v$ along a streamline of a curved flow in a horizontal plane (Fig. 16-38).

(a) Show that $dp/dr = \rho v^2/r$, so that the pressure increases by an amount $\rho v^2/r$ per unit distance perpendicular to the streamline as we go from the concave to the convex side of the streamline.

(b) Then use Bernoulli's equation and this result to show that $vr$ equals a constant, so that speeds increase toward the center of curvature. Hence streamlines that are uniformly spaced in a straight pipe will be crowded toward the inner wall of a curved passage and widely spaced toward the outer wall. This prediction should be compared with Problem 12 of Chapter 15 in which the curved motion is produced by rotating a container. There the speed varied directly with $r$, but here it varies inversely.

(c) Show that this flow is irrotational.

**solution**
- (a) Radial force balance on fluid element gives
  $$\frac{dp}{dr}=\rho\frac{v^2}{r}. $$
- (b) Along neighboring streamlines at same elevation:
  $$dp+\rho v\,dv=0.$$
  Combine with (a):
  $$\rho\frac{v^2}{r}+\rho v\frac{dv}{dr}=0
  \Rightarrow \frac{d(vr)}{dr}=0
  \Rightarrow vr=\text{const}. $$
- (c) For planar swirl with $v_\theta=K/r$, $v_r=0$:
  $$\omega_z=\frac{1}{r}\frac{d(r v_\theta)}{dr}=\frac{1}{r}\frac{dK}{dr}=0,$$
  so the flow is irrotational (except possible singularity at $r=0$).

---

### T10 (16.14)
A fluid of viscosity $\eta$ flows steadily through a horizontal cylindrical pipe of radius $R$ and length $L$, as shown in Fig. 16-41.

(a) Consider an arbitrary cylinder of fluid of radius $r$. Show that the viscous force $F$ due to the neighboring layer is

$$
F = -\eta(2\pi rL)\frac{dv}{dr}.
$$

(b) Show that the force $F'$ pushing that cylinder of fluid through the pipe is

$$
F' = (\pi r^2)\Delta p.
$$

(c) Use the equilibrium condition to obtain an expression for $dv$ in terms of $dr$. Integrate the expression to obtain Eq. 16-18.

**solution**
- Use force balance on radius-$r$ fluid cylinder in steady flow: driving pressure force and viscous shear balance.
- From $F+F'=0$ and given (a)(b), integrate with boundary condition $v(R)=0$.
- Velocity profile:
  $$v(r)=\frac{\Delta p}{4\eta L}(R^2-r^2).$$
- This is Poiseuille parabolic profile (Eq. 16-18).

---

### T11 (16.17)
Consider a uniform U-tube with a diaphragm at the bottom and filled with a liquid to different heights in each arm (see Fig. 16-31). Now imagine that the diaphragm is punctured so that the liquid flows from left to right.

(a) Show that the application of Bernoulli's equation to points 1 and 3 leads to a contradiction.

(b) Explain why Bernoulli's equation is not applicable here. (Hint: Is the flow steady?)

**solution**
- (a) If Bernoulli were (incorrectly) applied between free surfaces 1 and 3:
  $$p_1=p_3=p_{atm},\quad v_1\approx v_3\approx0$$
  would imply $z_1=z_3$, contradicting the actual unequal liquid heights.
- (b) Reason: the flow after puncture is unsteady (time-dependent) and includes dissipation; simple steady Bernoulli along a streamline is not valid for this situation.

---

### T12 (16.21)
A hollow tube has a disk $DD$ attached to its end (Fig. 16-33). When air of density $\rho$ is blown through the tube, the disk attracts the card $CC$. Let the area of the card be $A$ and let $v$ be the average air speed between the card and the disk. Calculate the resultant upward force on $CC$. Neglect the card's weight; assume that $v_0 \ll v$, where $v_0$ is the air speed in the hollow tube.

**solution**
- Air speed in the gap is large ($v$), so pressure there is reduced.
- With $v_0\ll v$, Bernoulli gives pressure drop approximately
  $$\Delta p \approx \frac{1}{2}\rho v^2.$$
- Pressure below card is about ambient, above card (in gap) is lower by $\Delta p$.
- Net upward force on card:
  $$F_{up}=\Delta p\,A\approx\frac{1}{2}\rho A v^2.$$
