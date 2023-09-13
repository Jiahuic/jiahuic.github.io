---
layout: page
permalink: /teaching/math-3583/4_dimensional_analysis/
title: Dimensional Analysis
---
### **4 Dimensional Analysis**
#### **4.1 Dimensional Reduction**
The concept of a fundamental dimension for physical variables such as force, density, and velocity
can be broken down into length $$L$$, time $$T$$, and mass $$M$$.
Moreover, length, time, and mass are independent in the sense that one of them cannot be written in terms of the other two.
For problems involving thermodynamics,  we need to include
temperature $$\theta$$ and for electrical problem, we need to include current $$I$$.
These give the *fundamental dimensions*.

**Dimensions Notation.** *Given a physical quantity q, the fundamental dimensions of a will be denoted as $$\lbrack\lbrack q \rbrack\rbrack$$.
In the case of when $$q$$ is dimensionless, $$\lbrack\lbrack q \rbrack\rbrack=1$$.*

For example, $$\lbrack\lbrack v_0 \rbrack\rbrack=L/T$$, $$\lbrack\lbrack x \rbrack\rbrack=L$$ and $$\lbrack\lbrack x_M/R \rbrack\rbrack=1$$.

|Quantity|Dimensions| |Quantity|Dimensions|
|---|---||---|---|
|Acceleration| $$ LT^{-2} $$ ||Angle| $$ \text{Dimensionless} $$ |
|Angular Acceleration| $$ T^{-2} $$ ||Angular Momentum| $$ L^2MT^{-1} $$ |
|Angular Velocity| $$ T^{-1} $$ ||Area| $$ L^2 $$ |
|Energy/Work| $$ L^2MT^{-2} $$ ||Force| $$ LMT^{-2} $$ |
|Frequency| $$ T^{-1} $$ ||Concentration| $$ L^{-3}M $$ |
|Length| $$ L $$ ||Mass| $$ M $$ |
|Mass Density| $$ L^{-3}M $$ ||Momentum| $$ LMT^{-1} $$ |
|Power| $$ L^2MT^{-3} $$ ||Pressure/Stress| $$ L^{-1}MT^{-2} $$ |
|Elastic Modulus| $$ L^{-1}MT^{-2} $$ ||Surface Tension| $$ MT^{-2} $$ |
|Time| $$ T $$ ||Torque| $$ L^2MT^{-2} $$ |
|Velocity| $$ LT^{-1} $$ ||Viscosity (Dynamic)| $$ L^{-1}MT^{-1} $$ |
|Viscosity (Kinematic)| $$ L^2T^{-1} $$ ||Volume| $$ L^3 $$ |
|Wavelength| $$ L $$ ||Strain| $$ \text{Dimensionless} $$ |
|Enthalpy| $$ L^2MT^{-2} $$ ||Entropy| $$ L^2MT^{-2}\theta^{-1} $$ |
|Gas Constant| $$ L^2MT^{-2}\theta^{-1} $$ ||Internal Energy| $$ L^2MT^{-2} $$ |
|Specific Heat| $$ L^2T^{-2}\theta^{-1} $$ ||Temperature| $$ \theta $$ |
|Thermal Conductivity| $$ LMT^{-3}\theta^{-1} $$ ||Thermal Diffusivity| $$ L^2T^{-1} $$ |
|Heat Transfer Coefficient| $$ LMT^{-3}\theta^{-1} $$ ||Capacitance| $$ L^2M^{-1}T^4I^2 $$ |
|Charge| $$ IT $$ ||Charge Density| $$ L^{-3}IT $$ |
|Electrical Conductivity| $$ L^{-3}M^{-1}T^3I^2 $$ ||Admittance| $$ L^{-2}M^{-1}T^3I^2 $$ |
|Electric Potential/Voltage| $$ L^2MT^{-3}I^{-1} $$ ||Current Density| $$ L^{-2}IT $$ |
|Electric Current| $$ I $$ ||Electric Field Intensity| $$ LMT^{-3}I^{-1} $$ |
|Inductance| $$ L^2MT^{-2}I^{-2} $$ ||Magnetic Intensity| $$ L^{-1}IT $$ |
|Magnetic Flux Density| $$ MT^{-2}I^{-1} $$ ||Magnetic Permeability| $$ LMT^{-2}I^{-2} $$ |
|Electric Permittivity| $$ L^{-3}M^{-1}T^4I^2 $$ ||Electric Resistance| $$ L^2MT^{-3}I^{-2} $$ |

#### **4.2 Maximum Height of a Projectile**

##### **4.2.1 Using Calculus**
In classical physics, the motion of a projectile is often simplified using a constant gravitational field. However, a more accurate description involves accounting for the variation in the Earth's gravitational field with height. In this lecture, we'll explore the motion of a projectile under the influence of such a variable gravitational field, characterized by the equation:

$$
\frac{d^2 x}{dt^2} = -\frac{gR^2}{(R+x)^2}
$$

Here $$g$$ is the acceleration due to gravity near the surface of the Earth, $$R$$ is the Earth's radius, and $$x(t)$$ is the height of the projectile at time $$t$$.

Initial conditions are given as $$x(0) = 0$$ and $$\frac{dx}{dt}(0) = v_0$$.

**Assumption: $$R \gg x$$**
To make this problem more tractable, we'll use the assumption that the height $$x$$ of the projectile is much smaller than the Earth's radius $$R$$.

Under this assumption, we can Taylor-expand the denominator in the equation, obtaining:

$$
\frac{gR^2}{(R+x)^2} \approx \frac{gR^2}{R^2} = g
$$

This simplifies our equation to:

$$
\frac{d^2 x}{dt^2} = -g
$$

**Solving the Equation of Motion**
With this simplification, we now have a constant acceleration equation, which we can solve easily. Integrating once, we get:

$$
\frac{dx}{dt} = -gt + v_0
$$

Integrating a second time:

$$
x(t) = -\frac{1}{2}gt^2 + v_0 t
$$

We know that at maximum height, $$\frac{dx}{dt} = 0$$. Using this condition:

$$
0 = -gt + v_0  \implies t = \frac{v_0}{g}
$$

Substituting this into $$x(t)$$:

$$
x_{\text{max}} = -\frac{1}{2}g\left(\frac{v_0}{g}\right)^2 + v_0 \frac{v_0}{g}
$$

This simplifies to:

$$
x_{\text{max}} = \frac{v_0^2}{2g}
$$

##### **4.1.2 Using Dimensional Analysis**
Dimensional analysis is a powerful tool for solving physical problems, particularly when a straightforward analytical solution is elusive. The technique relies on the principle that every physical quantity must have consistent dimensions across an equation. We'll use this technique to find an expression for the maximum height ($$x_M$$) of a projectile launched vertically, given the initial velocity ($$v_0$$) and the gravitational acceleration ($$g$$).

**Hypothesis: Dimensional Homogeneity**
We start with the hypothesis that the maximum height ($$x_M$$) of the projectile is a function of $$v_0$$ and $$g$$:

$$
[[ x_M ]] = [[ m^a v_0^b g^c ]]
$$

Here, $$a, b, c$$ are unknown exponents that we aim to find.

**Fundamental Dimensions**
- $$L$$: Length
- $$M$$: Mass
- $$T$$: Time

**Dimensional Analysis**
We express $$x_M$$, $$v_0$$, and $$g$$ in terms of the fundamental dimensions:

- $$x_M$$ has dimensions $$L$$
- $$v_0$$ has dimensions $$L T^{-1}$$
- $$g$$ has dimensions $$L T^{-2}$$

Our hypothesis becomes:

$$
[[ L ]] = [[ M^a (L T^{-1})^b (L T^{-2})^c ]]
$$

Expanding this out, we get:

$$
L = M^a L^{b+c} T^{-b-2c}
$$

Comparing the exponents for each dimension, we get the following equations:

1. For $$L$$: $$1 = b + c$$
2. For $$M$$: $$0 = a$$
3. For $$T$$: $$0 = -b - 2c$$

Solving these equations, we find:

- $$a = 0$$,
- $$b = 2$$,
- $$c = -1$$.

Substituting these back into the original equation, we get:

$$
x_M = \alpha\frac{v_0^2}{g}
$$

#### **4.3 Drag on a Sphere**

