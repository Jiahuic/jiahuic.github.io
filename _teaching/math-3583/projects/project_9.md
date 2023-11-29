---
layout: page
permalink: /teaching/math-3583/project_9/
title: "Project 9: Numerical Solution of the 1D Hyperbolic Equation"
---
#### Objective

To numerically solve the 1D first order hyperbolic equation and analyze the behavior of hyperbolic propagation under various initial and boundary conditions.

#### Problem Description

The 1D first order hyperbolic equation is given by:

$$
\frac{\partial u}{\partial t} + a \frac{\partial u}{\partial x} = 0
$$

where $$u(x,t)$$ is the displacement at position $$0\le x \le 3$$ and time $$t$$, and $$a=1$$ is the wave speed.
Here, you try to solve the periodic boundary conditions and the the initial condition is given by:
$$
g(x) = \begin{cases}
1, \frac{1}{4} \leq x \leq \frac{3}{4} \\
1-|4x-6|, \frac{5}{4} \leq x \leq \frac{7}{4} \\
\cos^2\pi(2x-5), \frac{9}{4} \leq x \leq \frac{11}{4} \\
0, \text{otherwise}
\end{cases}
$$

#### Project Steps
##### Step 1: Discretization
Discretize the domain $$0\le x \le 3$$ into $$N$$ equally spaced points $$x_i$$ with $$i=0,1,2,\cdots,N$$ and $$x_0=0$$ and $$x_N=3$$. The spacing between two adjacent points is $$\Delta x = 3/N$$. The time domain $$0\le t \le T$$ is also discretized into $$M$$ equally spaced points $$t_j$$ with $$j=0,1,2,\cdots,M$$ and $$t_0=0$$ and $$t_M=T$$. The spacing between two adjacent points is $$\Delta t = T/M$$.
```python
import numpy as np
N = 100
M = 100
x = np.linspace(0,3,N+1)
dx = 3/N
t = np.linspace(0,10,M+1)
dt = 10/M
```
##### Step 2: Initial Condition
Compute the initial condition $$u(x,0)=g(x)$$ at $$t=0$$ using the given function $$g(x)$$ in Python.
```python
u0 = np.zeros(N+1)
for i in range(N+1):
    if 1/4 <= x[i] <= 3/4:
        u0[i] = 1
    elif 5/4 <= x[i] <= 7/4:
        u0[i] = 1 - abs(4*x[i]-6)
    elif 9/4 <= x[i] <= 11/4:
        u0[i] = np.cos(np.pi*(2*x[i]-5))**2
```
##### Step 3: Boundary Condition
The periodic boundary condition is $$u(0,t)=u(3,t)$$ for all $$t$$.
```python
u0[0] = u0[N]
```
##### Step 4: Numerical Solution
The explicit Forward Time Centered Space difference equation of the Wave Equation is,

$$
\frac{u^{n+1}_{j}-u^{n}_{j}}{\Delta_t}+\big(\frac{u^n_{j+1}-u^n_{j-1}}{2\Delta_x}\big)=0.
$$

Rearranging the equation we get,

$$
u_{j}^{n+1}=u^{n}_{j}-\lambda a(u_{j+1}^{n}-u_{j-1}^{n}),
$$

for $$j=0,...,10$$ where $$\lambda=\frac{\Delta_t}{2\Delta_x}$$.

This gives the formula for the unknown term $$w^{n+1}_{j}$$ at the $$(j,n+1)$$ mesh points
in terms of $$x[j]$$ along the $$n$$-th time row.

Use the forward difference scheme to discretize the 1D first order hyperbolic equation and obtain the numerical solution $$u(x,t)$$ at $$t=t_j$$ for $$j=1,2,\cdots,M$$.
```python
u = np.zeros((M+1,N+1))
lambda = dt/dx
u[0,:] = u0
for j in range(1,M+1):
    for i in range(1,N+1):
        u[j,i] = u[j-1,i] - u[j-1,i]*(u[j-1,i]-u[j-1,i-1])*lambda
```
##### Step 5: Plot the Numerical Solution
Plot the numerical solution $$u(x,t)$$ at $$t=t_j$$ for $$j=1,2,\cdots,M$$.
The plot is like a flash movie, which shows the propagation of the wave.
```python
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111)
ax.set_xlim(0,3)
ax.set_ylim(0,2)
ax.set_xlabel('x')
ax.set_ylabel('u')
ax.set_title('Numerical Solution')

from matplotlib import animation
```

##### Step 6: Question: what if use smaller time step?

#### Deliverables
Have a jupyter-notebook to show your code and results.
Based on the results, prepare a 4-minute presentation to show your results and answer the question.

**Reference**: [Wave Equation](https://john-s-butler-dit.github.io/NumericalAnalysisBook/Chapter%2010%20-%20Hyperbolic%20Equations/1001_Wave%20Equation.html)
