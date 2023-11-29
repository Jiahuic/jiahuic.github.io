---
layout: page
permalink: /teaching/math-3583/project_2/
title: "Project 2: Dimensional Analysis of the Linearized KdV Equation"
---

The problem involves finding the velocity $$u (x, t)$$ of the waves in shallow water.
It is assumed to satisfy the linearized KdV equation,
which is 
$$
\frac{\partial u}{\partial t} + K \frac{\partial^3 u}{\partial x^3} = 0 \text{,   for } -\infty < x < \infty \text{, } 0< t,
$$
where $$K$$ is a positive constant.
The boundary conditions are $$u=0$$ as $$ x\rightarrow \pm \infty $$, and instead of an initial condition, assume the solution satisfies
$$
\int_{-\infty}^{\infty} u d x = \gamma, \forall t > 0.
$$
1. What are the dimensions of $$K$$ and  $$\gamma $$?
2. What four physical quantities does $$u(x,t)$$ depend on? With this, write down the equivalent version of $$ [[ u ]] = [[ x^at^bD^c (u_0)^d ]] $$ for this problem, and use the exponent $$d$$ for $$ \gamma $$.
3. The key observation is that if you increase $$\gamma$$ by a factor $$\alpha$$, then the solution $$u$$ increases by this same factor. To prove this, let $$U(x,t)$$ by the solution for a given $$\gamma$$. Setting $$u=\alpha U(x,t)$$, what problem does $$u$$ satisfy? Assuming the solution is unique, explain why this proves the key observation.
4. Explain why the key observation made in part 3. means that $$d=1$$. With this, find the general dimensionally reduced form of the solution. As usual, let $$\eta$$ denote the similarity variable.
5. Using the result from part 4., transform the KdV equation into an ordinary differential equation. How do the boundary conditions transform?
6. Show that the third order differential equation in part 5. can be simplified to one that is second order. You can assume $$F^{\prime\prime}\rightarrow 0$$ and $$\eta F \rightarrow 0$$ as $$\eta \rightarrow \pm \infty$$. As a hint, you might want to look for the expression $$(\eta F)^\prime$$ in your equation.
7. Rewrite the initial condition in terms of $$F$$.
8. The solution of the problem is $$F(\eta) = \text{Ai} (\eta)$$, where Ai is known as the Airy function. This function plays a central role in wave problems, both in fluids, electromagnetics, and quantum mechanics. Read [the history of Airy function](https://www.worldscientific.com/doi/suppl/10.1142/p345/suppl_file/p345_chap01.pdf).
